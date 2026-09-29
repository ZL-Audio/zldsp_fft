#if defined(ZLDSP_FFT_COMMON_MATH_HPP_) == defined(HWY_TARGET_TOGGLE)
#ifdef ZLDSP_FFT_COMMON_MATH_HPP_
#undef ZLDSP_FFT_COMMON_MATH_HPP_
#else
#define ZLDSP_FFT_COMMON_MATH_HPP_
#endif

#include <cmath>
#include <numbers>

#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();

namespace zldsp::fft::HWY_NAMESPACE::common::math {
    [[nodiscard]] inline double cospi(const double x) noexcept {
#if defined(__APPLE__)
        return __cospi(x);
#else
        return std::cos(x * std::numbers::pi_v<double>);
#endif
    }

    [[nodiscard]] inline double sinpi(const double x) noexcept {
#if defined(__APPLE__)
        return __sinpi(x);
#else
        return std::sin(x * std::numbers::pi_v<double>);
#endif
    }
}

HWY_AFTER_NAMESPACE();

#endif
