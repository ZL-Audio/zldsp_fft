#pragma once

#include <array>
#include <complex>
#include <cstddef>

#include "zldsp_fft_rfft-inl.hpp"

namespace zldsp::fft {
    template <typename F>
    class RFFT {
        using Complex = std::complex<F>;

        HWY_STATIC_NAMESPACE::RFFT<F> impl_;

    public:
        explicit RFFT(const size_t rfft_order) : impl_(rfft_order) {
        }

        [[nodiscard]] size_t get_size() const noexcept {
            return impl_.get_size();
        }

        [[nodiscard]] size_t get_order() const noexcept {
            return impl_.get_order();
        }

        /**
         * perform forward RFFT from real input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(F* in_buffer, Complex* out_buffer) noexcept {
            impl_.forward(in_buffer, out_buffer);
        }

        /**
         * perform forward RFFT from real input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(F* in_buffer, std::array<F*, 2> out_buffer) noexcept {
            impl_.forward(in_buffer, out_buffer);
        }

        /**
         * perform backward RFFT from AoS input to real output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(Complex* in_buffer, F* out_buffer) noexcept {
            impl_.backward(in_buffer, out_buffer);
        }

        /**
         * perform backward RFFT from SoA input to real output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::array<F*, 2> in_buffer, F* out_buffer) noexcept {
            impl_.backward(in_buffer, out_buffer);
        }

        /**
         * perform forward RFFT and write squared magnitudes
         * @param in_buffer
         * @param out_buffer
         */
        void forward_sqr_mag(F* in_buffer, F* out_buffer) noexcept {
            impl_.forward_sqr_mag(in_buffer, out_buffer);
        }

        /**
         * perform RFFT convolution with AoS spectrum
         * @param in_buffer
         * @param spectrum
         * @param out_buffer
         */
        void convolve(const F* in_buffer, const Complex* spectrum, F* out_buffer) noexcept {
            impl_.convolve(in_buffer, spectrum, out_buffer);
        }

        /**
         * perform RFFT convolution with SoA spectrum
         * @param in_buffer
         * @param spectrum
         * @param out_buffer
         */
        void convolve(const F* in_buffer, std::array<const F*, 2> spectrum, F* out_buffer) noexcept {
            impl_.convolve(in_buffer, spectrum, out_buffer);
        }

        /**
         * perform RFFT convolution with real spectrum
         * @param in_buffer
         * @param spectrum
         * @param out_buffer
         */
        void convolve(const F* in_buffer, const F* spectrum, F* out_buffer) noexcept {
            impl_.convolve(in_buffer, spectrum, out_buffer);
        }
    };
}
