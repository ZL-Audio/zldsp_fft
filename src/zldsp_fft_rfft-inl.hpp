#if defined(ZLDSP_FFT_RFFT_INL_HPP_) == defined(HWY_TARGET_TOGGLE)
#ifdef ZLDSP_FFT_RFFT_INL_HPP_
#undef ZLDSP_FFT_RFFT_INL_HPP_
#else
#define ZLDSP_FFT_RFFT_INL_HPP_
#endif

#include <array>
#include <complex>
#include <cstddef>
#include <type_traits>

#include <hwy/highway.h>

#include "common/zldsp_fft_common_execute.hpp"
#include "common/zldsp_fft_common_init.hpp"

HWY_BEFORE_NAMESPACE();

namespace zldsp::fft::HWY_NAMESPACE {
    template <typename F>
    class RFFT {
        static_assert(std::is_same_v<F, float> || std::is_same_v<F, double>,
                      "zldsp::fft::RFFT supports float and double");
        static_assert(!HWY_HAVE_SCALABLE, "zldsp::fft requires a fixed-width Highway target");

        using TargetTag = hwy::HWY_NAMESPACE::ScalableTag<F>;
        static constexpr size_t kTargetVectorBytes = TargetTag{}.MaxBytes();
        static_assert(kTargetVectorBytes == 16 || kTargetVectorBytes == 32,
                      "zldsp::fft requires a 128-bit or 256-bit Highway target");

        using Complex = std::complex<F>;

    private:
        common::CFFTState<F> state_;
        size_t rfft_size_;
        hwy::AlignedFreeUniquePtr<F[]> rfft_w_;
        hwy::AlignedFreeUniquePtr<F[]> rfft_workspace_;
        common::SoAPtr<F> forward_cfft_output_;
        common::SoAPtr<F> backward_cfft_input_;

    public:
        explicit RFFT(const size_t rfft_order) : rfft_size_(static_cast<size_t>(1) << rfft_order) {
            state_.cfft_order = rfft_order - 1;
            state_.cfft_size = static_cast<size_t>(1) << state_.cfft_order;
            common::init_cfft_state(state_.cfft_order, state_);
            if (state_.cfft_order < 6) {
                state_.workspace = common::allocate_aligned<F>(4 * state_.cfft_size);
                forward_cfft_output_ = common::make_soa<F>({
                    state_.workspace.get(),
                    state_.workspace.get() + state_.cfft_size,
                });
                backward_cfft_input_ = forward_cfft_output_;
            } else if (state_.num_macro_stages == 0) {
                if (state_.micro_stages.size() % 2 == 0) {
                    forward_cfft_output_ = common::make_soa<F>({
                        state_.workspace.get(),
                        state_.workspace.get() + state_.micro_stride,
                    });
                } else {
                    forward_cfft_output_ = common::make_soa<F>({
                        state_.workspace.get() + 2 * state_.micro_stride,
                        state_.workspace.get() + 3 * state_.micro_stride,
                    });
                }
                backward_cfft_input_ = common::make_soa<F>({
                    state_.workspace.get(),
                    state_.workspace.get() + state_.micro_stride,
                });
            } else {
                const size_t rfft_workspace_stride = state_.cfft_size + common::get_cache_color_padding<F>();
                rfft_workspace_ = common::allocate_aligned<F>(2 * rfft_workspace_stride);
                forward_cfft_output_ = common::make_soa<F>({
                    rfft_workspace_.get(),
                    rfft_workspace_.get() + rfft_workspace_stride,
                });
                backward_cfft_input_ = forward_cfft_output_;
            }
            common::generate_rfft_pre_post_w(state_.cfft_order, rfft_w_);
        }

        [[nodiscard]] size_t get_size() const noexcept {
            return rfft_size_;
        }

        [[nodiscard]] size_t get_order() const noexcept {
            return state_.cfft_order + 1;
        }

        /**
         * perform forward RFFT from real input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(F* in_buffer, Complex* out_buffer) noexcept {
            execute_forward(in_buffer, common::make_aos(out_buffer));
        }

        /**
         * perform forward RFFT from real input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(F* in_buffer, std::array<F*, 2> out_buffer) noexcept {
            execute_forward(in_buffer, common::make_soa(out_buffer));
        }

        /**
         * perform backward RFFT from AoS input to real output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(Complex* in_buffer, F* out_buffer) noexcept {
            execute_backward(common::make_aos(in_buffer), out_buffer);
        }

        /**
         * perform backward RFFT from SoA input to real output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::array<F*, 2> in_buffer, F* out_buffer) noexcept {
            execute_backward(common::make_soa(in_buffer), out_buffer);
        }

        /**
         * perform forward RFFT and write squared magnitudes
         * @param in_buffer
         * @param out_buffer
         */
        void forward_sqr_mag(F* in_buffer, F* out_buffer) noexcept {
            common::execute_cfft<true>(state_, common::make_aos(reinterpret_cast<Complex*>(in_buffer)),
                                       forward_cfft_output_);
            common::execute_rfft_forward_sqr_mag_post(forward_cfft_output_, out_buffer,
                                                      state_.cfft_order, rfft_w_.get());
        }

    private:
        /**
         * execute forward RFFT
         * @tparam OutPtr
         * @param in
         * @param out
         */
        template <typename OutPtr>
        void execute_forward(F* in, OutPtr out) noexcept {
            common::execute_cfft<true>(state_, common::make_aos(reinterpret_cast<Complex*>(in)),
                                       forward_cfft_output_);
            common::execute_rfft_forward_post(forward_cfft_output_, out,
                                              state_.cfft_order, rfft_w_.get());
        }

        /**
         * execute backward RFFT
         * @tparam InPtr
         * @param in
         * @param out
         */
        template <typename InPtr>
        void execute_backward(InPtr in, F* out) noexcept {
            common::execute_rfft_backward_pre(in, backward_cfft_input_,
                                              state_.cfft_order, rfft_w_.get());
            common::execute_cfft<false>(state_, backward_cfft_input_,
                                        common::make_aos<F>(reinterpret_cast<Complex*>(out)));
        }
    };
}

HWY_AFTER_NAMESPACE();

#endif
