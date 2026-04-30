#pragma once

#include <hwy/highway.h>

namespace zldsp::fft::common {
    namespace hn = hwy::HWY_NAMESPACE;

    template <typename F>
    struct AoSPtr {
        F* HWY_RESTRICT comp;

        [[nodiscard]] HWY_INLINE constexpr AoSPtr shift(const size_t offset) const {
            return AoSPtr{comp + offset};
        }

        [[nodiscard]] static constexpr size_t get_complex_offset(const size_t offset) {
            return (offset << 1);
        }
    };

    template <typename F>
    struct SoAPtr {
        F* HWY_RESTRICT real;
        F* HWY_RESTRICT imag;

        [[nodiscard]] HWY_INLINE constexpr SoAPtr shift(const size_t offset) const {
            return SoAPtr{real + offset, imag + offset};
        }

        [[nodiscard]] static constexpr size_t get_complex_offset(const size_t offset) {
            return offset;
        }
    };

    template <bool is_forward, class D, typename Ptr, typename V>
    HWY_INLINE void load_complex(D d, Ptr ptr, V& r, V& i) {
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
                hn::LoadInterleaved2(d, ptr.comp, r, i);
            } else {
                hn::LoadInterleaved2(d, ptr.comp, i, r);
            }
        }
    }

    template <bool is_forward, class D, typename Ptr, typename V>
    HWY_INLINE void store_complex(D d, Ptr ptr, const V r, const V i) {
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
                hn::StoreInterleaved2(r, i, d, ptr.comp);
            } else {
                hn::StoreInterleaved2(i, r, d, ptr.comp);
            }
        }
    }

    template <bool is_forward, typename F, typename Ptr>
    HWY_INLINE void load_scalar(Ptr ptr, F& r, F& i) {
        if constexpr (std::is_same_v<Ptr, SoAPtr<F>>) {
            r = is_forward ? ptr.real[0] : ptr.imag[0];
            i = is_forward ? ptr.imag[0] : ptr.real[0];
        } else {
            r = is_forward ? ptr.comp[0] : ptr.comp[1];
            i = is_forward ? ptr.comp[1] : ptr.comp[0];
        }
    }

    template <bool is_forward, typename F, typename Ptr>
    HWY_INLINE void store_scalar(Ptr ptr, const F r, const F i) {
        if constexpr (std::is_same_v<Ptr, SoAPtr<F>>) {
            ptr.real[0] = is_forward ? r : i;
            ptr.imag[0] = is_forward ? i : r;
        } else {
            ptr.comp[0] = is_forward ? r : i;
            ptr.comp[1] = is_forward ? i : r;
        }
    }

    template <typename F>
    static common::AoSPtr<F> make_aos(std::complex<F>* ptr) {
        return common::AoSPtr<F>{reinterpret_cast<F*>(ptr)};
    }

    template <typename F>
    static common::SoAPtr<F> make_soa(std::array<F*, 2> ptr) {
        return common::SoAPtr<F>{ptr[0], ptr[1]};
    }
}