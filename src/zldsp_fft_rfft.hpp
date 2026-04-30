#pragma once

#include <array>

#include "zldsp_fft_common_execute.hpp"

namespace zldsp::fft {
    namespace hn = hwy::HWY_NAMESPACE;

    template <typename F>
    class RFFT {
        using C = std::complex<F>;

    private:
        size_t rfft_size_;
        size_t cfft_order_;
        hwy::AlignedFreeUniquePtr<F[]> workspace_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_;
        hwy::AlignedFreeUniquePtr<F[]> rfft_twiddles_;
        std::vector<size_t> twiddles_shift_;
        std::vector<common::StageType> stages_;

    public:
        explicit RFFT(const size_t rfft_order) :
            rfft_size_(1 << rfft_order),
            cfft_order_(rfft_order - 1) {
            common::init_cfft_state(cfft_order_, stages_,
                                    twiddles_shift_, twiddles_, workspace_);
            if (cfft_order_ < 6) {
                const size_t cfft_size = 1 << cfft_order_;
                workspace_ = hwy::AllocateAligned<F>(4 * common::get_stride<F>(cfft_size));
            }
            common::generate_rfft_pre_post_twiddles(cfft_order_, rfft_twiddles_);
        }

        [[nodiscard]] size_t get_size() const {
            return rfft_size_;
        }

        [[nodiscard]] size_t get_order() const {
            return cfft_order_ + 1;
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

    private:
        /**
         * execute forward RFFT
         * @tparam OutPtr
         * @param in_ptr
         * @param out_ptr
         */
        template <typename OutPtr>
        void execute_forward(F* in_ptr, OutPtr out_ptr) {
            const size_t cfft_size = 1 << cfft_order_;
            const size_t stride = common::get_stride<F>(cfft_size);

            F* cfft_out = workspace_.get();
            if (cfft_order_ > 5) {
                cfft_out = ((stages_.size() - 1) % 2 == 1) ? workspace_.get() : workspace_.get() + 2 * stride;
            }

            const auto temp_soa = common::make_soa<F>({cfft_out, cfft_out + stride});

            common::execute_cfft<true>(cfft_order_, workspace_.get(),
                twiddles_.get(), twiddles_shift_, stages_,
                common::make_aos(reinterpret_cast<C*>(in_ptr)), temp_soa);

            common::execute_rfft_forward_post(cfft_order_, rfft_twiddles_.get(),
                temp_soa, out_ptr);
        }

        /**
         * execute backward RFFT
         * @tparam InPtr
         * @param in_ptr
         * @param out_ptr
         */
        template <typename InPtr>
        void execute_backward(InPtr in_ptr, F* out_ptr) {
            const size_t cfft_size = 1 << cfft_order_;
            const size_t stride = common::get_stride<F>(cfft_size);

            F* cfft_in = workspace_.get();
            const auto temp_soa = common::make_soa<F>({cfft_in, cfft_in + stride});

            common::execute_rfft_backward_pre(cfft_order_, rfft_twiddles_.get(),
                in_ptr, temp_soa);

            common::execute_cfft<false>(cfft_order_, workspace_.get(),
                twiddles_.get(), twiddles_shift_, stages_,
                temp_soa, common::make_aos(reinterpret_cast<C*>(out_ptr)));
        }
    };
}
