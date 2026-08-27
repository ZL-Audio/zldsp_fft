#pragma once

#include "zldsp_fft_cfft-inl.hpp"

namespace zldsp::fft {
    template <typename F>
    class CFFT {
        using Complex = std::complex<F>;

        HWY_STATIC_NAMESPACE::CFFT<F> impl_;

    public:
        explicit CFFT(const size_t cfft_order) : impl_(cfft_order) {}

        [[nodiscard]] size_t get_size() const {
            return impl_.get_size();
        }

        [[nodiscard]] size_t get_order() const {
            return impl_.get_order();
        }

        /**
         * perform forward CFFT from AoS input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(Complex* in_buffer, Complex* out_buffer) noexcept {
            impl_.forward(in_buffer, out_buffer);
        }

        /**
         * perform backward CFFT from AoS input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(Complex* in_buffer, Complex* out_buffer) noexcept {
            impl_.backward(in_buffer, out_buffer);
        }

        /**
         * perform forward CFFT from AoS input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(Complex* in_buffer, std::array<F*, 2> out_buffer) noexcept {
            impl_.forward(in_buffer, out_buffer);
        }

        /**
         * perform backward CFFT from AoS input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(Complex* in_buffer, std::array<F*, 2> out_buffer) noexcept {
            impl_.backward(in_buffer, out_buffer);
        }

        /**
         * perform forward CFFT from SoA input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(std::array<F*, 2> in_buffer, Complex* out_buffer) noexcept {
            impl_.forward(in_buffer, out_buffer);
        }

        /**
         * perform backward CFFT from SoA input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::array<F*, 2> in_buffer, Complex* out_buffer) noexcept {
            impl_.backward(in_buffer, out_buffer);
        }

        /**
         * perform forward CFFT from SoA input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(std::array<F*, 2> in_buffer, std::array<F*, 2> out_buffer) noexcept {
            impl_.forward(in_buffer, out_buffer);
        }

        /**
         * perform backward CFFT from SoA input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::array<F*, 2> in_buffer, std::array<F*, 2> out_buffer) noexcept {
            impl_.backward(in_buffer, out_buffer);
        }
    };
}
