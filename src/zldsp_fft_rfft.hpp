#pragma once

#include <array>

#include "common/zldsp_fft_common_init.hpp"
#include "common/zldsp_fft_common_execute.hpp"

namespace zldsp::fft {
    namespace hn = hwy::HWY_NAMESPACE;

    template <typename F>
    class RFFT {
        using C = std::complex<F>;

    private:
        common::CFFTState<F> state_;
        size_t rfft_size_;
        hwy::AlignedFreeUniquePtr<F[]> rfft_twiddles_;
        hwy::AlignedFreeUniquePtr<F[]> rfft_workspace_;
        common::SoAPtr<F> forward_temp_soa_;
        common::SoAPtr<F> backward_temp_soa_;

    public:
        explicit RFFT(const size_t rfft_order) :
            rfft_size_(1 << rfft_order) {
            state_.cfft_order = rfft_order - 1;
            state_.cfft_size = 1 << state_.cfft_order;
            common::init_cfft_state(state_.cfft_order, state_);
            if (state_.cfft_order < 6) {
                state_.workspace = hwy::AllocateAligned<F>(4 * state_.cfft_size);
                forward_temp_soa_ = common::make_soa<F>({state_.workspace.get(),
                                                         state_.workspace.get() + state_.cfft_size});
                backward_temp_soa_ = forward_temp_soa_;
            } else {
                if (state_.num_macro_stages == 0) {
                    if (state_.micro_stages.size() % 2 == 0) {
                        forward_temp_soa_ = common::make_soa<F>({state_.workspace.get(),
                                                                 state_.workspace.get() + state_.micro_stride});
                    } else {
                        forward_temp_soa_ = common::make_soa<F>({state_.workspace.get() + 2 * state_.micro_stride,
                                                                 state_.workspace.get() + 3 * state_.micro_stride});
                    }
                    backward_temp_soa_ = common::make_soa<F>({state_.workspace.get(),
                                                              state_.workspace.get() + state_.micro_stride});
                } else {
                    rfft_workspace_ = hwy::AllocateAligned<F>(2 * state_.cfft_size);
                    forward_temp_soa_ = common::make_soa<F>({rfft_workspace_.get(),
                                                             rfft_workspace_.get() + state_.cfft_size});
                    backward_temp_soa_ = forward_temp_soa_;
                }
            }
            common::generate_rfft_pre_post_twiddles(state_.cfft_order, rfft_twiddles_);
        }

        [[nodiscard]] size_t get_size() const {
            return rfft_size_;
        }

        [[nodiscard]] size_t get_order() const {
            return state_.cfft_order + 1;
        }

        /**
         * real to AoS forward
         * @param in_buffer
         * @param out_buffer
         */
        void forward(F* in_buffer, C* out_buffer) {
            execute_forward(in_buffer, common::make_aos(out_buffer));
        }

        /**
         * real to SoA forward
         * @param in_buffer
         * @param out_buffer
         */
        void forward(F* in_buffer, std::array<F*, 2> out_buffer) {
            execute_forward(in_buffer, common::make_soa(out_buffer));
        }

        /**
         * AoS to real backward
         * @param in_buffer
         * @param out_buffer
         */
        void backward(C* in_buffer, F* out_buffer) {
            execute_backward(common::make_aos(in_buffer), out_buffer);
        }

        /**
         * SoA to real backward
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::array<F*, 2> in_buffer, F* out_buffer) {
            execute_backward(common::make_soa(in_buffer), out_buffer);
        }

        /**
         * real to squared magnitude forward
         * @param in_buffer
         * @param out_buffer
         */
        void forward_sqr_mag(F* in_buffer, F* out_buffer) {
            common::execute_cfft<true>(
                state_, common::make_aos(reinterpret_cast<C*>(in_buffer)), forward_temp_soa_);
            common::execute_rfft_forward_sqr_mag_post(
                state_.cfft_order, rfft_twiddles_.get(), forward_temp_soa_, out_buffer);
        }

    private:
        /**
         * execute forward RFFT
         * @tparam OutPtr
         * @param in_ptr
         * @param out_ptr
         */
        template <typename OutPtr>
        void execute_forward(F* in_ptr, OutPtr out_ptr) {
            common::execute_cfft<true>(
                state_, common::make_aos(reinterpret_cast<C*>(in_ptr)), forward_temp_soa_);
            common::execute_rfft_forward_post(
                state_.cfft_order, rfft_twiddles_.get(), forward_temp_soa_, out_ptr);
        }

        /**
         * execute backward RFFT
         * @tparam InPtr
         * @param in_ptr
         * @param out_ptr
         */
        template <typename InPtr>
        void execute_backward(InPtr in_ptr, F* out_ptr) {
            common::execute_rfft_backward_pre(
                state_.cfft_order, rfft_twiddles_.get(), in_ptr, backward_temp_soa_);
            common::execute_cfft<false>(
                state_, backward_temp_soa_, common::make_aos<F>(reinterpret_cast<C*>(out_ptr)));
        }
    };
}
