#if defined(ZLDSP_FFT_CFFT_INL_HPP_) == defined(HWY_TARGET_TOGGLE)
#ifdef ZLDSP_FFT_CFFT_INL_HPP_
#undef ZLDSP_FFT_CFFT_INL_HPP_
#else
#define ZLDSP_FFT_CFFT_INL_HPP_
#endif

#include <array>
#include <complex>
#include <cstddef>
#include <type_traits>

#include <hwy/highway.h>

#include "common/zldsp_fft_common_init.hpp"
#include "common/zldsp_fft_common_execute.hpp"

HWY_BEFORE_NAMESPACE();

namespace zldsp::fft::HWY_NAMESPACE {
    template <typename F>
    class CFFT {
        static_assert(std::is_same_v<F, float> || std::is_same_v<F, double>,
                      "zldsp::fft::CFFT supports float and double");
        static_assert(!HWY_HAVE_SCALABLE,
                      "zldsp::fft requires a fixed-width Highway target");

        using TargetTag = hwy::HWY_NAMESPACE::ScalableTag<F>;
        static constexpr size_t kTargetVectorBytes = TargetTag{}.MaxBytes();
        static_assert(kTargetVectorBytes == 16 || kTargetVectorBytes == 32,
                      "zldsp::fft requires a 128-bit or 256-bit Highway target");

        using Complex = std::complex<F>;

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
         * perform forward CFFT from AoS input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(Complex* in_buffer, Complex* out_buffer) noexcept {
            execute<true>(common::make_aos(in_buffer), common::make_aos(out_buffer));
        }

        /**
         * perform backward CFFT from AoS input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(Complex* in_buffer, Complex* out_buffer) noexcept {
            execute<false>(common::make_aos(in_buffer), common::make_aos(out_buffer));
        }

        /**
         * perform forward CFFT from AoS input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(Complex* in_buffer, std::array<F*, 2> out_buffer) noexcept {
            execute<true>(common::make_aos(in_buffer), common::make_soa(out_buffer));
        }

        /**
         * perform backward CFFT from AoS input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(Complex* in_buffer, std::array<F*, 2> out_buffer) noexcept {
            execute<false>(common::make_aos(in_buffer), common::make_soa(out_buffer));
        }

        /**
         * perform forward CFFT from SoA input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(std::array<F*, 2> in_buffer, Complex* out_buffer) noexcept {
            execute<true>(common::make_soa(in_buffer), common::make_aos(out_buffer));
        }

        /**
         * perform backward CFFT from SoA input to AoS output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::array<F*, 2> in_buffer, Complex* out_buffer) noexcept {
            execute<false>(common::make_soa(in_buffer), common::make_aos(out_buffer));
        }

        /**
         * perform forward CFFT from SoA input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void forward(std::array<F*, 2> in_buffer, std::array<F*, 2> out_buffer) noexcept {
            execute<true>(common::make_soa(in_buffer), common::make_soa(out_buffer));
        }

        /**
         * perform backward CFFT from SoA input to SoA output
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::array<F*, 2> in_buffer, std::array<F*, 2> out_buffer) noexcept {
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
        void execute(InPtr in_ptr, OutPtr out_ptr) noexcept {
            common::execute_cfft<is_forward>(state_, in_ptr, out_ptr);
        }
    };
}

HWY_AFTER_NAMESPACE();

#endif
