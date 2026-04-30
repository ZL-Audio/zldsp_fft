#pragma once

#include <cmath>
#include <numbers>

namespace zldsp::fft::common::math {
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
