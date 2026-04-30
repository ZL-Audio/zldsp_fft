#pragma once

#include <hwy/aligned_allocator.h>
#include <hwy/highway.h>
#include <cmath>
#include <numeric>
#include <vector>

#include "zldsp_fft_common_math.hpp"

namespace zldsp::fft::common {
    namespace hn = hwy::HWY_NAMESPACE;

    enum class StageType {
        kRadix8FirstPass,
        kRadix4FirstPass,
        kRadix4Width4,
        kRadix4,
        kRadix4LastPass,
    };

    /**
     * get padded size of a given CFFT size
     * @tparam F
     * @param cfft_size
     * @return
     */
    template <typename F>
    inline constexpr size_t get_stride(const size_t cfft_size) {
        return cfft_size + (64 / sizeof(F)) + 16;
    }

    /**
     * generate twiddles for order = 4 or order = 5
     * @tparam F
     * @param order
     * @param twiddles
     */
    template <typename F>
    inline void generate_order_4_5_twiddles(const size_t order,
                                            hwy::AlignedFreeUniquePtr<F[]>& twiddles) {
        const size_t num_twiddles = (order == 4 ? 60 : 120);

        twiddles = hwy::AllocateAligned<F>(num_twiddles << 1);
        F* HWY_RESTRICT twiddles_r = twiddles.get();
        F* HWY_RESTRICT twiddles_i = twiddles.get() + num_twiddles;

        size_t offset = 0;
        size_t width = (order == 4 ? 4 : 8);
        for (size_t i = 0; i < 2; ++i) {
            const double phase_step = -2.0 / static_cast<double>(width << 2);
            for (int mul = 1; mul < 4; ++mul) {
                const auto step = phase_step * static_cast<double>(mul);
                for (size_t k = 0; k < width; ++k, ++offset) {
                    const double phase = static_cast<double>(k) * step;
                    twiddles_r[offset] = static_cast<F>(math::cospi(phase));
                    twiddles_i[offset] = static_cast<F>(math::sinpi(phase));
                }
            }
            width = width << 2;
        }
    }

    /**
     * generates twiddles for order > 5
     * @tparam F
     * @param stages
     * @param twiddles_shift
     * @param twiddles
     */
    template <typename F>
    inline void generate_general_twiddles(std::vector<StageType>& stages,
                                          std::vector<size_t>& twiddles_shift,
                                          hwy::AlignedFreeUniquePtr<F[]>& twiddles) {
        static constexpr size_t lanes = hn::Lanes(hn::ScalableTag<F>());
        static constexpr size_t width4_vec = std::max(static_cast<size_t>(4), lanes);
        // calculate twiddle shift for each stage
        {
            size_t width = (stages[0] == StageType::kRadix4FirstPass) ? 4 : 8;
            for (size_t i = 1; i < stages.size(); ++i) {
                const auto stage = stages[i];
                if (stage == StageType::kRadix4Width4) {
                    twiddles_shift[i] = 6 * width4_vec;
                    width = width << 2;
                } else if (stage == StageType::kRadix4 || stage == StageType::kRadix4LastPass) {
                    const size_t num_blocks = std::max<size_t>(1, width / lanes);
                    twiddles_shift[i] = num_blocks * 6 * lanes;
                    width = width << 2;
                }
            }
        }
        // allocate twiddle
        {
            const auto num_twiddles =
                std::accumulate(twiddles_shift.begin(), twiddles_shift.end(), static_cast<size_t>(0));
            twiddles = hwy::AllocateAligned<F>(num_twiddles);
        }
        // calculate twiddle values
        {
            size_t offset = 0;
            size_t width = (stages[0] == StageType::kRadix4FirstPass) ? 4 : 8;
            for (size_t i = 1; i < stages.size(); ++i) {
                const auto stage = stages[i];
                if (stage == StageType::kRadix4Width4) {
                    const double phase_step = -2.0 / static_cast<double>(width << 2);
                    for (size_t l = 0; l < width4_vec; ++l) {
                        const auto phase = static_cast<double>(l % 4) * phase_step;
                        static constexpr int muls[3] = {1, 2, 3};
                        for (size_t m = 0; m < 3; ++m) {
                            const auto a = phase * static_cast<double>(muls[m]);
                            twiddles[offset + 2 * m * width4_vec + l] = static_cast<F>(math::cospi(a));
                            twiddles[offset + (2 * m + 1) * width4_vec + l] = static_cast<F>(math::sinpi(a));
                        }
                    }
                    offset += twiddles_shift[i];
                    width = width << 2;
                } else if (stage == StageType::kRadix4 || stage == StageType::kRadix4LastPass) {
                    const size_t num_blocks = std::max<size_t>(1, width / lanes);
                    const double phase_step = -2.0 / static_cast<double>(width << 2);
                    for (size_t b = 0; b < num_blocks; ++b) {
                        for (size_t l = 0; l < lanes; ++l) {
                            const size_t idx = (b * lanes + l) % width;
                            const auto phase = static_cast<double>(idx) * phase_step;
                            static constexpr int muls[3] = {1, 2, 3};
                            for (size_t m = 0; m < 3; ++m) {
                                const auto a = phase * static_cast<double>(muls[m]);
                                twiddles[offset + 2 * m * lanes + l] = static_cast<F>(math::cospi(a));
                                twiddles[offset + (2 * m + 1) * lanes + l] = static_cast<F>(math::sinpi(a));
                            }
                        }
                        offset += 6 * lanes;
                    }
                    width = width << 2;
                }
            }
        }
    }

    /**
     * init twiddles and working-space for CFFT
     * @tparam F
     * @param order
     * @param stages
     * @param twiddles_shift
     * @param twiddles
     * @param workspace
     */
    template <typename F>
    inline void init_cfft_state(
        const size_t order,
        std::vector<StageType>& stages,
        std::vector<size_t>& twiddles_shift,
        hwy::AlignedFreeUniquePtr<F[]>& twiddles,
        hwy::AlignedFreeUniquePtr<F[]>& workspace) {
        if (order >= 4 && order < 6) {
            common::generate_order_4_5_twiddles(order, twiddles);
        }

        if (order >= 6) {
            const auto mod_result = order % 2;
            if (mod_result == 1) {
                stages.emplace_back(StageType::kRadix8FirstPass);
                for (size_t i = 3; i < order - 2; i += 2) {
                    stages.emplace_back(StageType::kRadix4);
                }
            } else {
                stages.emplace_back(StageType::kRadix4FirstPass);
                stages.emplace_back(StageType::kRadix4Width4);
                for (size_t i = 4; i < order - 2; i += 2) {
                    stages.emplace_back(StageType::kRadix4);
                }
            }
            stages.emplace_back(StageType::kRadix4LastPass);

            twiddles_shift.resize(stages.size());
            twiddles_shift[0] = 0;

            common::generate_general_twiddles(stages, twiddles_shift, twiddles);

            const auto n = static_cast<size_t>(1) << order;
            workspace = hwy::AllocateAligned<F>(4 * get_stride<F>(n));
        }
    }

    /**
     * generate RFFT pre/post processing twiddles
     * @tparam F
     * @param cfft_order
     * @param rfft_twiddles
     */
    template <typename F>
    inline void generate_rfft_pre_post_twiddles(const size_t cfft_order,
                                                hwy::AlignedFreeUniquePtr<F[]>& rfft_twiddles) {
        size_t lanes = hn::Lanes(hn::ScalableTag<F>());
        if (cfft_order == 0 || cfft_order == 1) {
            const auto d = hn::CappedTag<F, 1>();
            lanes = hn::Lanes(d);
        } else if (cfft_order == 2) {
            const auto d = hn::CappedTag<F, 2>();
            lanes = hn::Lanes(d);
        } else if (cfft_order == 3) {
            const auto d = hn::CappedTag<F, 4>();
            lanes = hn::Lanes(d);
        } else if (cfft_order == 4) {
            const auto d = hn::CappedTag<F, 8>();
            lanes = hn::Lanes(d);
        }
        const size_t cfft_size = static_cast<size_t>(1) << cfft_order;
        const size_t rfft_size = cfft_size * 2;
        const size_t num_elements = cfft_size / 2;
        if (num_elements == 0) {
            return;
        }
        const size_t num_blocks = (num_elements + lanes - 1) / lanes;
        const size_t twiddle_size = num_blocks * lanes * 2;
        rfft_twiddles = hwy::AllocateAligned<F>(twiddle_size);

        const double phase_step = 2.0 / static_cast<double>(rfft_size);
        for (size_t b = 0; b < num_blocks; ++b) {
            for (size_t l = 0; l < lanes; ++l) {
                const size_t idx = b * lanes + l + 1;
                if (idx <= num_elements) {
                    const auto a = static_cast<double>(idx) * phase_step;
                    rfft_twiddles[b * lanes * 2 + l] = static_cast<F>(0.5 * math::cospi(a));
                    rfft_twiddles[b * lanes * 2 + lanes + l] = static_cast<F>(0.5 * math::sinpi(a) + 0.5);
                } else {
                    rfft_twiddles[b * lanes * 2 + l] = static_cast<F>(0.0);
                    rfft_twiddles[b * lanes * 2 + lanes + l] = static_cast<F>(0.5);
                }
            }
        }
    }
}
