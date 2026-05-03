#pragma once

#include <array>

#include "common/zldsp_fft_common_init.hpp"
#include "common/zldsp_fft_common_execute.hpp"

namespace zldsp::fft {
    namespace hn = hwy::HWY_NAMESPACE;

    template <typename F>
    class CFFT {
        using C = std::complex<F>;

    private:
        common::CFFTState<F> state_;

    public:
        explicit CFFT(const size_t cfft_order) {
            state_.cfft_size = 1 << cfft_order;
            state_.cfft_order = cfft_order;
            common::init_cfft_state(cfft_order, state_);
        }

        [[nodiscard]] size_t get_size() const {
            return state_.cfft_size;
        }

        [[nodiscard]] size_t get_order() const {
            return state_.cfft_order;
        }

        /**
         * AoS to AoS forward
         * @param in_buffer
         * @param out_buffer
         */
        void forward(C* in_buffer, C* out_buffer) {
            execute<true>(common::make_aos(in_buffer), common::make_aos(out_buffer));
        }

        /**
         * AoS to AoS backward
         * @param in_buffer
         * @param out_buffer
         */
        void backward(C* in_buffer, C* out_buffer) {
            execute<false>(common::make_aos(in_buffer), common::make_aos(out_buffer));
        }

        /**
         * AoS to SoA forward
         * @param in_buffer
         * @param out_buffer
         */
        void forward(C* in_buffer, std::array<F*, 2> out_buffer) {
            execute<true>(common::make_aos(in_buffer), common::make_soa(out_buffer));
        }

        /**
         * AoS to SoA backward
         * @param in_buffer
         * @param out_buffer
         */
        void backward(C* in_buffer, std::array<F*, 2> out_buffer) {
            execute<false>(common::make_aos(in_buffer), common::make_soa(out_buffer));
        }

        /**
         * SoA to AoS forward
         * @param in_buffer
         * @param out_buffer
         */
        void forward(std::array<F*, 2> in_buffer, C* out_buffer) {
            execute<true>(common::make_soa(in_buffer), common::make_aos(out_buffer));
        }

        /**
         * SoA to AoS backward
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::array<F*, 2> in_buffer, C* out_buffer) {
            execute<false>(common::make_soa(in_buffer), common::make_aos(out_buffer));
        }

        /**
         * SoA to SoA forward
         * @param in_buffer
         * @param out_buffer
         */
        void forward(std::array<F*, 2> in_buffer, std::array<F*, 2> out_buffer) {
            execute<true>(common::make_soa(in_buffer), common::make_soa(out_buffer));
        }

        /**
         * SoA to SoA backward
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::array<F*, 2> in_buffer, std::array<F*, 2> out_buffer) {
            execute<false>(common::make_soa(in_buffer), common::make_soa(out_buffer));
        }

    private:
        /**
         * execute CFFT
         * @tparam is_forward
         * @tparam InPtr
         * @tparam OutPtr
         * @param in_ptr
         * @param out_ptr
         */
        template <bool is_forward, typename InPtr, typename OutPtr>
        void execute(InPtr in_ptr, OutPtr out_ptr) {
            common::execute_cfft<is_forward>(state_, in_ptr, out_ptr);
        }
    };
}
