#pragma once

#include <complex>
#include <hwy/highway.h>

namespace zldsp::fft::common {
    namespace hn = hwy::HWY_NAMESPACE;

    template <typename F>
    struct AoSPtr {
        F* HWY_RESTRICT interleaved;

        [[nodiscard]] HWY_INLINE constexpr AoSPtr shift(const size_t offset) const noexcept {
            return AoSPtr{interleaved + offset};
        }

        [[nodiscard]] static constexpr size_t get_complex_offset(const size_t offset) noexcept {
            return (offset << 1);
        }
    };

    template <typename F>
    struct SoAPtr {
        F* HWY_RESTRICT real;
        F* HWY_RESTRICT imag;

        [[nodiscard]] HWY_INLINE constexpr SoAPtr shift(const size_t offset) const noexcept {
            return SoAPtr{real + offset, imag + offset};
        }

        [[nodiscard]] static constexpr size_t get_complex_offset(const size_t offset) noexcept {
            return offset;
        }
    };

    /**
     * load complex numbers from a pointer wrapper into SIMD registers
     * @tparam is_forward
     * @tparam D
     * @tparam Ptr
     * @tparam V
     * @param d
     * @param ptr
     * @param r
     * @param i
     */
    template <bool is_forward, class D, typename Ptr, typename V>
    HWY_INLINE void load_complex(D d, Ptr ptr, V& r, V& i) noexcept {
        using F = hn::TFromD<D>;
        if constexpr (std::is_same_v<Ptr, SoAPtr<F>>) {
            if constexpr (is_forward) {
                r = hn::LoadU(d, ptr.real);
                i = hn::LoadU(d, ptr.imag);
            } else {
                i = hn::LoadU(d, ptr.real);
                r = hn::LoadU(d, ptr.imag);
            }
        } else {
            if constexpr (is_forward) {
                hn::LoadInterleaved2(d, ptr.interleaved, r, i);
            } else {
                hn::LoadInterleaved2(d, ptr.interleaved, i, r);
            }
        }
    }

    /**
     * store complex numbers from SIMD registers through a pointer wrapper
     * @tparam is_forward
     * @tparam D
     * @tparam Ptr
     * @tparam V
     * @param d
     * @param ptr
     * @param r
     * @param i
     */
    template <bool is_forward, class D, typename Ptr, typename V>
    HWY_INLINE void store_complex(D d, Ptr ptr, const V r, const V i) noexcept {
        using F = hn::TFromD<D>;
        if constexpr (std::is_same_v<Ptr, SoAPtr<F>>) {
            if constexpr (is_forward) {
                hn::StoreU(r, d, ptr.real);
                hn::StoreU(i, d, ptr.imag);
            } else {
                hn::StoreU(i, d, ptr.real);
                hn::StoreU(r, d, ptr.imag);
            }
        } else {
            if constexpr (is_forward) {
                hn::StoreInterleaved2(r, i, d, ptr.interleaved);
            } else {
                hn::StoreInterleaved2(i, r, d, ptr.interleaved);
            }
        }
    }

    /**
     * load a complex number from a pointer wrapper into scalars
     * @tparam is_forward
     * @tparam F
     * @tparam Ptr
     * @param ptr
     * @param r
     * @param i
     */
    template <bool is_forward, typename F, typename Ptr>
    HWY_INLINE void load_scalar(Ptr ptr, F& r, F& i) noexcept {
        if constexpr (std::is_same_v<Ptr, SoAPtr<F>>) {
            r = is_forward ? ptr.real[0] : ptr.imag[0];
            i = is_forward ? ptr.imag[0] : ptr.real[0];
        } else {
            r = is_forward ? ptr.interleaved[0] : ptr.interleaved[1];
            i = is_forward ? ptr.interleaved[1] : ptr.interleaved[0];
        }
    }

    /**
     * store a complex number from scalars through a pointer wrapper
     * @tparam is_forward
     * @tparam F
     * @tparam Ptr
     * @param ptr
     * @param r
     * @param i
     */
    template <bool is_forward, typename F, typename Ptr>
    HWY_INLINE void store_scalar(Ptr ptr, const F r, const F i) noexcept {
        if constexpr (std::is_same_v<Ptr, SoAPtr<F>>) {
            ptr.real[0] = is_forward ? r : i;
            ptr.imag[0] = is_forward ? i : r;
        } else {
            ptr.interleaved[0] = is_forward ? r : i;
            ptr.interleaved[1] = is_forward ? i : r;
        }
    }

    /**
     * make an AoSPtr from a complex pointer
     * @tparam F
     * @param ptr
     * @return
     */
    template <typename F>
    static AoSPtr<F> make_aos(std::complex<F>* ptr) noexcept {
        return AoSPtr<F>{reinterpret_cast<F*>(ptr)};
    }

    /**
     * make a SoAPtr from two real/imag pointers
     * @tparam F
     * @param ptr
     * @return
     */
    template <typename F>
    static SoAPtr<F> make_soa(std::array<F*, 2> ptr) noexcept {
        return SoAPtr<F>{ptr[0], ptr[1]};
    }
}
