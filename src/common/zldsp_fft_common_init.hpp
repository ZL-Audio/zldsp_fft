#if defined(ZLDSP_FFT_COMMON_INIT_HPP_) == defined(HWY_TARGET_TOGGLE)
#ifdef ZLDSP_FFT_COMMON_INIT_HPP_
#undef ZLDSP_FFT_COMMON_INIT_HPP_
#else
#define ZLDSP_FFT_COMMON_INIT_HPP_
#endif

#include <algorithm>
#include <complex>
#include <cstddef>
#include <new>
#include <numeric>
#include <utility>
#include <vector>

#include <hwy/aligned_allocator.h>
#include <hwy/highway.h>

#include "zldsp_fft_common_math.hpp"
#include "zldsp_fft_common_system.hpp"

HWY_BEFORE_NAMESPACE();

namespace zldsp::fft::HWY_NAMESPACE::common {
    namespace hn = hwy::HWY_NAMESPACE;

    enum class StageType {
        kRadix8FirstPass,
        kRadix4FirstPass,
        kRadix4Width4,
        kRadix4,
        kRadix4LastPass,
    };

    template <typename T>
    [[nodiscard]] inline hwy::AlignedFreeUniquePtr<T[]> allocate_aligned(const size_t count) {
        auto ptr = hwy::AllocateAligned<T>(count);
        if (!ptr) {
            throw std::bad_alloc{};
        }
        return ptr;
    }

    template <typename F>
    struct CFFTState {
        size_t cfft_size = 0;
        size_t cfft_order = 0;
        hwy::AlignedFreeUniquePtr<F[]> workspace;

        size_t micro_order = 0;
        std::vector<StageType> micro_stages;
        hwy::AlignedFreeUniquePtr<F[]> micro_w;
        std::vector<size_t> micro_w_strides;
        size_t micro_stride = 0;

        size_t num_macro_stages = 0;
        hwy::AlignedFreeUniquePtr<F[]> macro_w;
        std::vector<size_t> macro_w_strides;
        size_t macro_stride = 0;

        size_t num_micro_ffts = 0;
        size_t transpose_tile_stride = 0;
        std::vector<size_t> radix4_digit_reversal;
    };

    inline constexpr size_t kHybridTransposeTileRows = 64;

    /**
     * get cache-line color padding while retaining Highway alignment
     * @tparam F
     * @return padding in scalar elements
     */
    template <typename F>
    [[nodiscard]] inline size_t get_cache_color_padding() {
        const size_t line_size = std::max<size_t>(get_cache_line_size(), HWY_ALIGNMENT);
        const size_t aligned_size = ((line_size + HWY_ALIGNMENT - 1) / HWY_ALIGNMENT) * HWY_ALIGNMENT;
        return aligned_size / sizeof(F);
    }

    /**
     * get padded workspace stride for one CFFT component
     * @tparam F
     * @param cfft_size
     * @return stride in scalar elements
    */
    template <typename F>
    [[nodiscard]] inline constexpr size_t get_cfft_stride(const size_t cfft_size) noexcept {
        return cfft_size + (64 / sizeof(F)) + 16;
    }

    /**
     * get padding for each terminal-transpose row
     * @tparam F
     * @return padding in scalar elements
    */
    template <typename F>
    [[nodiscard]] inline constexpr size_t get_transpose_padding() noexcept {
        return 64 / sizeof(std::complex<F>);
    }

    /**
     * generate twiddle factors for order = 4 or order = 5
     * @tparam F
     * @param order
     * @param w
     */
    template <typename F>
    inline void generate_order_4_5_w(const size_t order, hwy::AlignedFreeUniquePtr<F[]>& w) {
        const size_t w_size = (order == 4 ? 60 : 120);

        w = common::allocate_aligned<F>(w_size << 1);
        F* HWY_RESTRICT w_r = w.get();
        F* HWY_RESTRICT w_i = w.get() + w_size;

        size_t offset = 0;
        size_t width = (order == 4 ? 4 : 8);
        for (size_t i = 0; i < 2; ++i) {
            const double phase_step = -2.0 / static_cast<double>(width << 2);
            for (int mul = 1; mul < 4; ++mul) {
                const auto step = phase_step * static_cast<double>(mul);
                for (size_t k = 0; k < width; ++k, ++offset) {
                    const double phase = static_cast<double>(k) * step;
                    w_r[offset] = static_cast<F>(math::cospi(phase));
                    w_i[offset] = static_cast<F>(math::sinpi(phase));
                }
            }
            width <<= 2;
        }
    }

    /**
     * generate Stockham stage twiddle factors for order > 5
     * @tparam F
     * @param stages
     * @param w
     * @param w_strides
     */
    template <typename F>
    inline void generate_stockham_w(const std::vector<StageType>& stages, hwy::AlignedFreeUniquePtr<F[]>& w,
                                    std::vector<size_t>& w_strides) {
        static constexpr size_t lanes = hn::MaxLanes(hn::ScalableTag<F>());
        static constexpr size_t width4_size = std::max<size_t>(4, lanes);
        // calculate twiddle stride for each stage
        {
            size_t width = (stages[0] == StageType::kRadix4FirstPass) ? 4 : 8;
            for (size_t i = 1; i < stages.size(); ++i) {
                const auto stage = stages[i];
                if (stage == StageType::kRadix4Width4) {
                    w_strides[i] = 6 * width4_size;
                    width <<= 2;
                } else if (stage == StageType::kRadix4 || stage == StageType::kRadix4LastPass) {
                    const size_t num_blocks = std::max<size_t>(1, width / lanes);
                    w_strides[i] = num_blocks * 6 * lanes;
                    width <<= 2;
                }
            }
        }
        // allocate twiddle storage
        {
            const auto w_size = std::accumulate(w_strides.begin(), w_strides.end(), static_cast<size_t>(0));
            w = common::allocate_aligned<F>(w_size);
        }
        // calculate twiddle values
        {
            size_t offset = 0;
            size_t width = (stages[0] == StageType::kRadix4FirstPass) ? 4 : 8;
            for (size_t i = 1; i < stages.size(); ++i) {
                const auto stage = stages[i];
                if (stage == StageType::kRadix4Width4) {
                    const double phase_step = -2.0 / static_cast<double>(width << 2);
                    for (size_t l = 0; l < width4_size; ++l) {
                        const auto phase = static_cast<double>(l % 4) * phase_step;
                        static constexpr int kMultipliers[] = {1, 2, 3};
                        for (size_t m = 0; m < 3; ++m) {
                            const auto a = phase * static_cast<double>(kMultipliers[m]);
                            w[offset + 2 * m * width4_size + l] = static_cast<F>(math::cospi(a));
                            w[offset + (2 * m + 1) * width4_size + l] = static_cast<F>(math::sinpi(a));
                        }
                    }
                    offset += w_strides[i];
                    width <<= 2;
                } else if (stage == StageType::kRadix4 || stage == StageType::kRadix4LastPass) {
                    const size_t num_blocks = std::max<size_t>(1, width / lanes);
                    const double phase_step = -2.0 / static_cast<double>(width << 2);
                    for (size_t b = 0; b < num_blocks; ++b) {
                        for (size_t l = 0; l < lanes; ++l) {
                            const size_t index = (b * lanes + l) % width;
                            const auto phase = static_cast<double>(index) * phase_step;
                            static constexpr int kMultipliers[] = {1, 2, 3};
                            for (size_t m = 0; m < 3; ++m) {
                                const auto a = phase * static_cast<double>(kMultipliers[m]);
                                w[offset + 2 * m * lanes + l] = static_cast<F>(math::cospi(a));
                                w[offset + (2 * m + 1) * lanes + l] = static_cast<F>(math::sinpi(a));
                            }
                        }
                        offset += 6 * lanes;
                    }
                    width <<= 2;
                }
            }
        }
    }

    /**
     * initialize twiddle factors for Stockham DIT CFFT
     * @tparam F
     * @param order
     * @param stages
     * @param w
     * @param w_strides
     * @return stride size
     */
    template <typename F>
    inline size_t init_stockham_cfft_state(const size_t order, std::vector<StageType>& stages,
                                           hwy::AlignedFreeUniquePtr<F[]>& w, std::vector<size_t>& w_strides) {
        if (order < 4) {
            return 0;
        } else if (order < 6) {
            common::generate_order_4_5_w(order, w);
            return 0;
        } else {
            const bool is_odd_order = (order % 2) != 0;
            if (is_odd_order) {
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

            w_strides.resize(stages.size());
            w_strides[0] = 0;

            common::generate_stockham_w(stages, w, w_strides);

            return get_cfft_stride<F>(static_cast<size_t>(1) << order);
        }
    }

    /**
     * initialize twiddle factors for macro Cooley-Tukey DIF CFFT
     * @tparam F
     * @param order
     * @param num_stages
     * @param w
     * @param w_strides
     * @return stride size
     */
    template <typename F>
    inline size_t init_cooley_tukey_cfft_state(const size_t order, const size_t num_stages,
                                               hwy::AlignedFreeUniquePtr<F[]>& w,
                                               std::vector<size_t>& w_strides) {
        static constexpr size_t lanes = hn::MaxLanes(hn::ScalableTag<F>());
        const size_t n = static_cast<size_t>(1) << order;
        w_strides.resize(num_stages);
        // calculate twiddle stride for each stage
        {
            for (size_t i = 0; i < num_stages; ++i) {
                const size_t sub_n = n >> (2 * i);
                const size_t width = sub_n >> 2;
                const size_t num_blocks = std::max<size_t>(1, width / lanes);
                w_strides[i] = num_blocks * 6 * lanes;
            }
        }
        // allocate twiddle storage
        {
            const auto w_size = std::accumulate(w_strides.begin(), w_strides.end(), static_cast<size_t>(0));
            w = common::allocate_aligned<F>(w_size);
        }
        // calculate twiddle values
        {
            size_t offset = 0;
            for (size_t i = 0; i < num_stages; ++i) {
                const size_t sub_n = n >> (2 * i);
                const size_t width = sub_n >> 2;
                const size_t num_blocks = std::max<size_t>(1, width / lanes);
                const double phase_step = -2.0 / static_cast<double>(sub_n);
                for (size_t b = 0; b < num_blocks; ++b) {
                    for (size_t l = 0; l < lanes; ++l) {
                        const size_t index = (b * lanes + l) % width;
                        const auto phase = static_cast<double>(index) * phase_step;
                        static constexpr int kMultipliers[] = {1, 2, 3};
                        for (size_t m = 0; m < 3; ++m) {
                            const auto a = phase * static_cast<double>(kMultipliers[m]);
                            w[offset + 2 * m * lanes + l] = static_cast<F>(math::cospi(a));
                            w[offset + (2 * m + 1) * lanes + l] = static_cast<F>(math::sinpi(a));
                        }
                    }
                    offset += 6 * lanes;
                }
            }
        }
        return get_cfft_stride<F>(static_cast<size_t>(1) << order);
    }

    /**
     * initialize radix-4 digit-reversal table
     * @param num_micro_ffts
     * @param num_macro_stages
     * @param radix4_digit_reversal
     */
    inline void init_radix4_digit_reversal(const size_t num_micro_ffts, const size_t num_macro_stages,
                                           std::vector<size_t>& radix4_digit_reversal) {
        radix4_digit_reversal.resize(num_micro_ffts);
        for (size_t i = 0; i < num_micro_ffts; ++i) {
            size_t reversed_index = 0;
            size_t index = i;
            for (size_t digit_index = 0; digit_index < num_macro_stages; ++digit_index) {
                const size_t digit = index & 3;
                index >>= 2;
                reversed_index = (reversed_index << 2) | digit;
            }
            radix4_digit_reversal[i] = reversed_index;
        }
    }

    /**
     * initialize twiddle factors and workspace for CFFT
     * @tparam F
     * @param cfft_order
     * @param state
     */
    template <typename F>
    inline void init_cfft_state(const size_t cfft_order, CFFTState<F>& state) {
        const auto [max_l1_order, hybrid_switch_order] = common::get_hybrid_order_thresholds<F>();
        if (cfft_order < hybrid_switch_order) {
            state.micro_order = cfft_order;
            state.num_macro_stages = 0;
            state.micro_stride = common::init_stockham_cfft_state(state.micro_order, state.micro_stages,
                                                                  state.micro_w, state.micro_w_strides);
            if (state.micro_stride > 0) {
                state.workspace = common::allocate_aligned<F>(4 * state.micro_stride);
            }
        } else {
            if ((cfft_order - max_l1_order) % 2 != 0) {
                state.micro_order = max_l1_order - 1;
            } else {
                state.micro_order = max_l1_order;
            }
            state.num_macro_stages = (cfft_order - state.micro_order) / 2;

            state.micro_stride = common::init_stockham_cfft_state(state.micro_order, state.micro_stages,
                                                                  state.micro_w, state.micro_w_strides);

            state.macro_stride = common::init_cooley_tukey_cfft_state(state.cfft_order, state.num_macro_stages,
                                                                      state.macro_w, state.macro_w_strides);

            state.num_micro_ffts = static_cast<size_t>(1) << (state.cfft_order - state.micro_order);
            const size_t micro_fft_size = static_cast<size_t>(1) << state.micro_order;
            const size_t transpose_tile_rows = std::min<size_t>(kHybridTransposeTileRows, state.num_micro_ffts);
            state.transpose_tile_stride =
                transpose_tile_rows * (micro_fft_size + get_transpose_padding<F>()) + get_cache_color_padding<F>();

            const size_t workspace_size =
                2 * state.macro_stride + 2 * state.transpose_tile_stride + 2 * state.micro_stride;
            state.workspace = common::allocate_aligned<F>(workspace_size);

            common::init_radix4_digit_reversal(state.num_micro_ffts, state.num_macro_stages,
                                               state.radix4_digit_reversal);
        }
    }

    /**
     * generate RFFT pre/post-processing twiddle factors
     * @tparam F
     * @param cfft_order
     * @param w
     */
    template <typename F>
    inline void generate_rfft_pre_post_w(const size_t cfft_order, hwy::AlignedFreeUniquePtr<F[]>& w) {
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
        const size_t rfft_size = cfft_size << 1;
        const size_t num_elements = cfft_size / 2;
        if (num_elements == 0) {
            return;
        }
        const size_t num_blocks = (num_elements + lanes - 1) / lanes;
        const size_t w_size = num_blocks * lanes * 2;
        w = common::allocate_aligned<F>(w_size);

        const double phase_step = 2.0 / static_cast<double>(rfft_size);
        for (size_t b = 0; b < num_blocks; ++b) {
            for (size_t l = 0; l < lanes; ++l) {
                const size_t index = b * lanes + l + 1;
                if (index <= num_elements) {
                    const auto a = static_cast<double>(index) * phase_step;
                    w[b * lanes * 2 + l] = static_cast<F>(0.5 * math::cospi(a));
                    w[b * lanes * 2 + lanes + l] = static_cast<F>(0.5 * math::sinpi(a) + 0.5);
                } else {
                    w[b * lanes * 2 + l] = static_cast<F>(0.0);
                    w[b * lanes * 2 + lanes + l] = static_cast<F>(0.5);
                }
            }
        }
    }
}

HWY_AFTER_NAMESPACE();

#endif
