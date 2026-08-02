#pragma once

#include <algorithm>
#include <bit>
#include <utility>
#include <vector>
#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace zldsp::fft::common {
    /**
     * get data-cache line size
     * @return data-cache line size in bytes
     */
    inline size_t get_cache_line_size() {
        size_t line_size = 64;
#if defined(__APPLE__)
        size_t size = sizeof(line_size);
        sysctlbyname("hw.cachelinesize", &line_size, &size, nullptr, 0);
#elif defined(__linux__) && defined(_SC_LEVEL1_DCACHE_LINESIZE)
        const long size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
        if (size > 0) {
            line_size = static_cast<size_t>(size);
        }
#elif defined(_WIN32)
        DWORD buffer_size = 0;
        GetLogicalProcessorInformation(nullptr, &buffer_size);
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
            std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
                buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
            if (GetLogicalProcessorInformation(buffer.data(), &buffer_size)) {
                for (const auto& info : buffer) {
                    if (info.Relationship == RelationCache && info.Cache.Level == 1 &&
                        (info.Cache.Type == CacheData || info.Cache.Type == CacheUnified)) {
                        line_size = info.Cache.LineSize;
                        break;
                    }
                }
            }
        }
#endif
        return line_size;
    }

    /**
     * get L1 data-cache size
     * @return size in bytes
     */
    inline size_t get_l1d_cache_size() {
        size_t l1d_size = 32768;
#if defined(__APPLE__)
        size_t size = sizeof(l1d_size);
        sysctlbyname("hw.l1dcachesize", &l1d_size, &size, nullptr, 0);
#elif defined(__linux__)
        long size = sysconf(_SC_LEVEL1_DCACHE_SIZE);
        if (size > 0) {
            l1d_size = static_cast<size_t>(size);
        }
#elif defined(_WIN32)
        DWORD buffer_size = 0;
        GetLogicalProcessorInformation(nullptr, &buffer_size);
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
            std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
                buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
            if (GetLogicalProcessorInformation(buffer.data(), &buffer_size)) {
                for (const auto& info : buffer) {
                    if (info.Relationship == RelationCache) {
                        if (info.Cache.Level == 1 && (info.Cache.Type == CacheData || info.Cache.Type ==
                            CacheUnified)) {
                            l1d_size = info.Cache.Size;
                            break;
                        }
                    }
                }
            }
        }
#endif
        return l1d_size;
    }

    /**
     * get L2 cache size
     * @return size in bytes
     */
    inline size_t get_l2_cache_size() {
        size_t l2_size = 262144;
#if defined(__APPLE__)
        size_t size = sizeof(l2_size);
        sysctlbyname("hw.l2cachesize", &l2_size, &size, nullptr, 0);
#elif defined(__linux__)
        long size = sysconf(_SC_LEVEL2_CACHE_SIZE);
        if (size > 0) {
            l2_size = static_cast<size_t>(size);
        }
#elif defined(_WIN32)
        DWORD buffer_size = 0;
        GetLogicalProcessorInformation(nullptr, &buffer_size);
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
            std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
                buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
            if (GetLogicalProcessorInformation(buffer.data(), &buffer_size)) {
                for (const auto& info : buffer) {
                    if (info.Relationship == RelationCache) {
                        if (info.Cache.Level == 2 &&
                            (info.Cache.Type == CacheData || info.Cache.Type == CacheUnified)) {
                            l2_size = info.Cache.Size;
                            break;
                        }
                    }
                }
            }
        }
#endif
        return l2_size;
    }

    /**
     * get cache-dependent order thresholds for the hybrid algorithm
     * @tparam F
     * @return {maximum order that can stay in L1 data cache, heuristic switching order for hybrid algorithm}
     */
    template <typename F>
    std::pair<size_t, size_t> get_hybrid_order_thresholds() {
        const size_t l1_size = get_l1d_cache_size();
        const size_t max_l1_elements = l1_size / (8 * sizeof(F));
        const size_t max_l1_order = (max_l1_elements == 0)
            ? static_cast<size_t>(0)
            : static_cast<size_t>(std::bit_width(max_l1_elements) - 1);

        const size_t l2_size = get_l2_cache_size();
        const size_t max_l2_elements = l2_size / (12 * sizeof(F));
        const size_t max_l2_order = (max_l2_elements == 0)
            ? static_cast<size_t>(0)
            : static_cast<size_t>(std::bit_width(max_l2_elements) - 1);

        const size_t hybrid_switch_order =
            std::max<size_t>(max_l1_order + 4, max_l2_order);

        return {max_l1_order, hybrid_switch_order};
    }
}
