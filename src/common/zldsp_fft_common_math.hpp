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
#if defined(__APPLE__)

    inline double cospi(const double x) {
        return __cospi(x);
    }

    inline double sinpi(const double x) {
        return __sinpi(x);
    }

#else

    inline double cospi(const double x) {
        return std::cos(x * std::numbers::pi_v<double>);
    }

    inline double sinpi(const double x) {
        return std::sin(x * std::numbers::pi_v<double>);
    }

#endif
}

HWY_AFTER_NAMESPACE();

#endif  // ZLDSP_FFT_COMMON_MATH_HPP_
