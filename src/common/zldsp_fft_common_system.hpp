#pragma once

#include <algorithm>
#include <bit>
#include <utility>
#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace zldsp::fft::common {
    /**
     *
     * @return L1 data cache size
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
     *
     * @return L2 cache size
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
     *
     * @return L3 cache size
     */
    inline size_t get_l3_cache_size() {
        size_t l3_size = 0;
#if defined(__APPLE__)
        l3_size = 0;
#elif defined(__linux__)
        long size = sysconf(_SC_LEVEL3_CACHE_SIZE);
        if (size > 0) {
            l3_size = static_cast<size_t>(size);
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
                        if (info.Cache.Level == 3 &&
                            (info.Cache.Type == CacheData || info.Cache.Type == CacheUnified)) {
                            l3_size = info.Cache.Size;
                            break;
                        }
                    }
                }
            }
        }
#endif
        return l3_size;
    }

    /**
     * get switching order at runtime
     * @tparam F
     * @return {maximum order that can stay in L1 data cache, heuristic switching order for hybrid algorithm}
     */
    template <typename F>
    std::pair<size_t, size_t> get_switch_order() {
        const size_t l1d = get_l1d_cache_size();
        const size_t max_l1_elements = l1d / (8 * sizeof(F));
        const size_t max_l1_order = (max_l1_elements == 0)
            ? static_cast<size_t>(0)
            : static_cast<size_t>(std::bit_width(max_l1_elements) - 1);

        const size_t l2d = get_l2_cache_size();
        const size_t max_l2_elements = l2d / (12 * sizeof(F));
        const size_t max_l2_order = (max_l2_elements == 0)
            ? static_cast<size_t>(0)
            : static_cast<size_t>(std::bit_width(max_l2_elements) - 1);

        const size_t l3d = get_l3_cache_size();
        const bool is_safe_l3 = (l3d > (64ULL * 1024 * 1024));
        size_t switch_order = std::max<size_t>(max_l1_order + 4, max_l2_order);
        if (is_safe_l3 && l3d > 0) {
            const size_t max_l3_elements = l3d / (12 * sizeof(F));
            const size_t max_l3_order = (max_l3_elements == 0)
                ? static_cast<size_t>(0)
                : static_cast<size_t>(std::bit_width(max_l3_elements) - 1);
            switch_order = std::max<size_t>(switch_order, max_l3_order);
        }

        return {max_l1_order, switch_order};
    }
}
