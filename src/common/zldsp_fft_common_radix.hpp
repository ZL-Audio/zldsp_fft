#pragma once

#include "zldsp_fft_common_init.hpp"
#include "zldsp_fft_common_structure.hpp"

namespace zldsp::fft::common {
    namespace hn = hwy::HWY_NAMESPACE;

    /**
     * perform a 4x4 matrix transpose
     * @tparam D
     * @tparam V
     * @param d
     * @param v0
     * @param v1
     * @param v2
     * @param v3
     * @param r0
     * @param r1
     * @param r2
     * @param r3
     */
    template <class D, class V>
    inline void transpose4x4(D d, V v0, V v1, V v2, V v3, V& r0, V& r1, V& r2, V& r3) {
        const size_t lanes = hn::Lanes(d);
        using T = hn::TFromD<D>;

        const auto t0 = hn::InterleaveLower(d, v0, v1);
        const auto t1 = hn::InterleaveLower(d, v2, v3);
        const auto t2 = hn::InterleaveUpper(d, v0, v1);
        const auto t3 = hn::InterleaveUpper(d, v2, v3);

        if constexpr (lanes == 2) {
            r0 = t0;
            r1 = t1;
            r2 = t2;
            r3 = t3;
        } else if constexpr (lanes == 4) {
            if constexpr (sizeof(T) == 8) {
                r0 = hn::ConcatLowerLower(d, t1, t0);
                r1 = hn::ConcatLowerLower(d, t3, t2);
                r2 = hn::ConcatUpperUpper(d, t1, t0);
                r3 = hn::ConcatUpperUpper(d, t3, t2);
            } else {
                hn::Repartition<uint64_t, D> d64;
                r0 = hn::BitCast(d, hn::InterleaveLower(d64, hn::BitCast(d64, t0), hn::BitCast(d64, t1)));
                r1 = hn::BitCast(d, hn::InterleaveUpper(d64, hn::BitCast(d64, t0), hn::BitCast(d64, t1)));
                r2 = hn::BitCast(d, hn::InterleaveLower(d64, hn::BitCast(d64, t2), hn::BitCast(d64, t3)));
                r3 = hn::BitCast(d, hn::InterleaveUpper(d64, hn::BitCast(d64, t2), hn::BitCast(d64, t3)));
            }
        } else if constexpr (lanes == 8) {
            hn::Repartition<uint64_t, D> d64;
            {
                const auto m0 = hn::BitCast(d, hn::InterleaveLower(d64, hn::BitCast(d64, t0), hn::BitCast(d64, t1)));
                const auto m1 = hn::BitCast(d, hn::InterleaveUpper(d64, hn::BitCast(d64, t0), hn::BitCast(d64, t1)));
                r0 = hn::ConcatLowerLower(d, m1, m0);
                r2 = hn::ConcatUpperUpper(d, m1, m0);
            }
            {
                const auto m2 = hn::BitCast(d, hn::InterleaveLower(d64, hn::BitCast(d64, t2), hn::BitCast(d64, t3)));
                const auto m3 = hn::BitCast(d, hn::InterleaveUpper(d64, hn::BitCast(d64, t2), hn::BitCast(d64, t3)));
                r1 = hn::ConcatLowerLower(d, m3, m2);
                r3 = hn::ConcatUpperUpper(d, m3, m2);
            }
        }
    }

    /**
     * perform a Stockham DIT radix-4 pass when width >= 8
     * @tparam F
     * @param in_aosoa
     * @param out_aosoa
     * @param n
     * @param width
     * @param w_ptr
     */
    template <typename F>
    inline void radix4_aosoa(const F* HWY_RESTRICT in_aosoa,
                             F* HWY_RESTRICT out_aosoa,
                             const size_t n, const size_t width, const F* HWY_RESTRICT w_ptr) {
        in_aosoa = static_cast<const F*>(HWY_ASSUME_ALIGNED(in_aosoa, HWY_ALIGNMENT));
        out_aosoa = static_cast<F*>(HWY_ASSUME_ALIGNED(out_aosoa, HWY_ALIGNMENT));
        w_ptr = static_cast<const F*>(HWY_ASSUME_ALIGNED(w_ptr, HWY_ALIGNMENT));

        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        const auto quarter_n = n >> 2;
        const auto half_n = n >> 1;
        const auto three_quarter_n = quarter_n + half_n;
        const auto three_over_two_n = three_quarter_n << 1;

        const auto double_width = width << 1;
        const auto triple_width = width * 3;
        const auto quad_width = width << 2;
        const auto sextuple_width = triple_width << 1;

        const size_t mask = width - 1;

        HWY_ASSUME(quarter_n >= lanes);
        HWY_ASSUME((quarter_n % lanes) == 0);
        for (size_t i = 0; i < quarter_n; i += lanes) {
            const F* HWY_RESTRICT in_shift = in_aosoa + (i << 1);
            const size_t k = i & mask;
            const F* HWY_RESTRICT w_shift = w_ptr + k * 6;

            hn::Vec<decltype(d)> s2_r, s2_i, s3_r, s3_i;
            {
                const auto w1_r = hn::Load(d, w_shift);
                const auto w1_i = hn::Load(d, w_shift + lanes);
                const auto r1 = hn::Load(d, in_shift + half_n);
                const auto i1 = hn::Load(d, in_shift + half_n + lanes);
                const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
                const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

                const auto w3_r = hn::Load(d, w_shift + lanes * 4);
                const auto w3_i = hn::Load(d, w_shift + lanes * 5);
                const auto r3 = hn::Load(d, in_shift + three_over_two_n);
                const auto i3 = hn::Load(d, in_shift + three_over_two_n + lanes);
                const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
                const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

                s2_r = hn::Add(t1_r, t3_r);
                s2_i = hn::Add(t1_i, t3_i);
                s3_r = hn::Sub(t1_r, t3_r);
                s3_i = hn::Sub(t1_i, t3_i);
            }

            hn::Vec<decltype(d)> s0_r, s0_i, s1_r, s1_i;

            const auto w2_r = hn::Load(d, w_shift + lanes * 2);
            const auto w2_i = hn::Load(d, w_shift + lanes * 3);
            const auto r2 = hn::Load(d, in_shift + n);
            const auto i2 = hn::Load(d, in_shift + n + lanes);
            const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
            const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

            const auto r0 = hn::Load(d, in_shift);
            const auto i0 = hn::Load(d, in_shift + lanes);

            s0_r = hn::Add(r0, t2_r);
            s0_i = hn::Add(i0, t2_i);
            s1_r = hn::Sub(r0, t2_r);
            s1_i = hn::Sub(i0, t2_i);

            const size_t j_times_4 = (i & ~mask) << 2;
            const size_t out_idx = j_times_4 + k;
            F* HWY_RESTRICT out_shift = out_aosoa + (out_idx << 1);

            hn::Store(hn::Add(s0_r, s2_r), d, out_shift);
            hn::Store(hn::Add(s0_i, s2_i), d, out_shift + lanes);
            hn::Store(hn::Add(s1_r, s3_i), d, out_shift + double_width);
            hn::Store(hn::Sub(s1_i, s3_r), d, out_shift + double_width + lanes);
            hn::Store(hn::Sub(s0_r, s2_r), d, out_shift + quad_width);
            hn::Store(hn::Sub(s0_i, s2_i), d, out_shift + quad_width + lanes);
            hn::Store(hn::Sub(s1_r, s3_i), d, out_shift + sextuple_width);
            hn::Store(hn::Add(s1_i, s3_r), d, out_shift + sextuple_width + lanes);
        }
    }

    /**
     * performs a Stockham DIT radix-4 first pass and convert data from AoS/SoA to AoSoA
     * @tparam is_forward
     * @tparam F
     * @tparam Ptr
     * @param in
     * @param out_aosoa
     * @param n
     */
    template <bool is_forward, typename F, typename Ptr>
    inline void radix4_first_pass_fused_aosoa(Ptr in,
                                              F* HWY_RESTRICT out_aosoa,
                                              const size_t n) {
        out_aosoa = static_cast<F*>(HWY_ASSUME_ALIGNED(out_aosoa, HWY_ALIGNMENT));

        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        const size_t quarter_n = n >> 2;
        const size_t in_offset1 = Ptr::get_complex_offset(n >> 2);
        const size_t in_offset2 = Ptr::get_complex_offset(n >> 1);
        const size_t in_offset3 = in_offset1 + in_offset2;

        HWY_ASSUME(quarter_n >= lanes);
        HWY_ASSUME((quarter_n % lanes) == 0);
        for (size_t j = 0; j < quarter_n; j += lanes) {
            const Ptr in_shift = in.shift(Ptr::get_complex_offset(j));
            hn::Vec<decltype(d)> t0_r, t0_i, t1_r, t1_i;
            {
                hn::Vec<decltype(d)> x0_r, x0_i, x2_r, x2_i;

                load_complex<is_forward>(d, in_shift, x0_r, x0_i);
                load_complex<is_forward>(d, in_shift.shift(in_offset2), x2_r, x2_i);

                t0_r = hn::Add(x0_r, x2_r);
                t0_i = hn::Add(x0_i, x2_i);
                t1_r = hn::Sub(x0_r, x2_r);
                t1_i = hn::Sub(x0_i, x2_i);
            }
            hn::Vec<decltype(d)> t2_r, t2_i, t3_r, t3_i;
            {
                hn::Vec<decltype(d)> x1_r, x1_i, x3_r, x3_i;
                load_complex<is_forward>(d, in_shift.shift(in_offset1), x1_r, x1_i);
                load_complex<is_forward>(d, in_shift.shift(in_offset3), x3_r, x3_i);

                t2_r = hn::Add(x1_r, x3_r);
                t2_i = hn::Add(x1_i, x3_i);
                t3_r = hn::Sub(x1_r, x3_r);
                t3_i = hn::Sub(x1_i, x3_i);
            }
            F* HWY_RESTRICT out_shift = out_aosoa + (j << 3);
            {
                const auto out0_r = hn::Add(t0_r, t2_r);
                const auto out2_r = hn::Sub(t0_r, t2_r);
                const auto out1_r = hn::Add(t1_r, t3_i);
                const auto out3_r = hn::Sub(t1_r, t3_i);

                hn::Vec<decltype(d)> r0, r1, r2, r3;
                transpose4x4(d, out0_r, out1_r, out2_r, out3_r, r0, r1, r2, r3);

                hn::Store(r0, d, out_shift);
                hn::Store(r1, d, out_shift + 2 * lanes);
                hn::Store(r2, d, out_shift + 4 * lanes);
                hn::Store(r3, d, out_shift + 6 * lanes);
            }
            {
                const auto out0_i = hn::Add(t0_i, t2_i);
                const auto out2_i = hn::Sub(t0_i, t2_i);
                const auto out1_i = hn::Sub(t1_i, t3_r);
                const auto out3_i = hn::Add(t1_i, t3_r);

                hn::Vec<decltype(d)> i0, i1, i2, i3;
                transpose4x4(d, out0_i, out1_i, out2_i, out3_i, i0, i1, i2, i3);

                hn::Store(i0, d, out_shift + lanes);
                hn::Store(i1, d, out_shift + 3 * lanes);
                hn::Store(i2, d, out_shift + 5 * lanes);
                hn::Store(i3, d, out_shift + 7 * lanes);
            }
        }
    }

    /**
     * perform a Stockham DIT radix-4 pass when width = 4
     * @tparam F
     * @param in_aosoa
     * @param out_aosoa
     * @param n
     * @param w_ptr
     */
    template <typename F>
    inline void radix4_width4_aosoa(const F* HWY_RESTRICT in_aosoa,
                                    F* HWY_RESTRICT out_aosoa,
                                    const size_t n, const F* HWY_RESTRICT w_ptr) {
        in_aosoa = static_cast<const F*>(HWY_ASSUME_ALIGNED(in_aosoa, HWY_ALIGNMENT));
        out_aosoa = static_cast<F*>(HWY_ASSUME_ALIGNED(out_aosoa, HWY_ALIGNMENT));
        w_ptr = static_cast<const F*>(HWY_ASSUME_ALIGNED(w_ptr, HWY_ALIGNMENT));

        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);
        static constexpr size_t step = (lanes > 4) ? lanes : 4;
        static constexpr size_t vecs_per_step = step / lanes;
        static constexpr size_t width4_vec = step;

        const size_t quarter_n = n >> 2;
        const size_t half_n = n >> 1;
        const size_t three_quarter_n = quarter_n * 3;
        const auto three_over_two_n = three_quarter_n << 1;

        hn::Vec<decltype(d)> w1_r_hoist, w1_i_hoist, w2_r_hoist, w2_i_hoist, w3_r_hoist, w3_i_hoist;
        if constexpr (vecs_per_step == 1) {
            w1_r_hoist = hn::Load(d, w_ptr);
            w1_i_hoist = hn::Load(d, w_ptr + width4_vec);
            w2_r_hoist = hn::Load(d, w_ptr + width4_vec * 2);
            w2_i_hoist = hn::Load(d, w_ptr + width4_vec * 3);
            w3_r_hoist = hn::Load(d, w_ptr + width4_vec * 4);
            w3_i_hoist = hn::Load(d, w_ptr + width4_vec * 5);
        }

        HWY_ASSUME(quarter_n >= step);
        HWY_ASSUME((quarter_n % step) == 0);
        for (size_t i = 0; i < quarter_n; i += step) {
            HWY_UNROLL(2)
            for (size_t v = 0; v < vecs_per_step; ++v) {
                const size_t vec_i = i + v * lanes;
                const F* HWY_RESTRICT in_shift = in_aosoa + (vec_i << 1);
                hn::Vec<decltype(d)> w1_r, w1_i, w2_r, w2_i, w3_r, w3_i;
                if constexpr (vecs_per_step == 1) {
                    w1_r = w1_r_hoist;
                    w1_i = w1_i_hoist;
                    w2_r = w2_r_hoist;
                    w2_i = w2_i_hoist;
                    w3_r = w3_r_hoist;
                    w3_i = w3_i_hoist;
                } else {
                    const size_t k = (v * lanes) & 3;
                    w1_r = hn::Load(d, w_ptr + k);
                    w1_i = hn::Load(d, w_ptr + k + width4_vec);
                    w2_r = hn::Load(d, w_ptr + k + width4_vec * 2);
                    w2_i = hn::Load(d, w_ptr + k + width4_vec * 3);
                    w3_r = hn::Load(d, w_ptr + k + width4_vec * 4);
                    w3_i = hn::Load(d, w_ptr + k + width4_vec * 5);
                }

                const auto r1 = hn::Load(d, in_shift + half_n);
                const auto i1 = hn::Load(d, in_shift + half_n + lanes);
                const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
                const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

                const auto r3 = hn::Load(d, in_shift + three_over_two_n);
                const auto i3 = hn::Load(d, in_shift + three_over_two_n + lanes);
                const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
                const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

                const auto s2_r = hn::Add(t1_r, t3_r);
                const auto s2_i = hn::Add(t1_i, t3_i);
                const auto s3_r = hn::Sub(t1_r, t3_r);
                const auto s3_i = hn::Sub(t1_i, t3_i);

                hn::Vec<decltype(d)> s0_r, s0_i, s1_r, s1_i;

                const auto r2 = hn::Load(d, in_shift + n);
                const auto i2 = hn::Load(d, in_shift + n + lanes);
                const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
                const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

                const auto r0 = hn::Load(d, in_shift);
                const auto i0 = hn::Load(d, in_shift + lanes);

                s0_r = hn::Add(r0, t2_r);
                s0_i = hn::Add(i0, t2_i);
                s1_r = hn::Sub(r0, t2_r);
                s1_i = hn::Sub(i0, t2_i);

                const auto out0_r = hn::Add(s0_r, s2_r);
                const auto out0_i = hn::Add(s0_i, s2_i);
                const auto out1_r = hn::Add(s1_r, s3_i);
                const auto out1_i = hn::Sub(s1_i, s3_r);
                const auto out2_r = hn::Sub(s0_r, s2_r);
                const auto out2_i = hn::Sub(s0_i, s2_i);
                const auto out3_r = hn::Sub(s1_r, s3_i);
                const auto out3_i = hn::Add(s1_i, s3_r);

                if constexpr (lanes > 4) {
                    F* HWY_RESTRICT out_shift = out_aosoa + (i << 3);

                    const auto out01_r_lo = hn::ConcatLowerLower(d, out1_r, out0_r);
                    const auto out01_i_lo = hn::ConcatLowerLower(d, out1_i, out0_i);
                    const auto out23_r_lo = hn::ConcatLowerLower(d, out3_r, out2_r);
                    const auto out23_i_lo = hn::ConcatLowerLower(d, out3_i, out2_i);

                    const auto out01_r_hi = hn::ConcatUpperUpper(d, out1_r, out0_r);
                    const auto out01_i_hi = hn::ConcatUpperUpper(d, out1_i, out0_i);
                    const auto out23_r_hi = hn::ConcatUpperUpper(d, out3_r, out2_r);
                    const auto out23_i_hi = hn::ConcatUpperUpper(d, out3_i, out2_i);

                    hn::Store(out01_r_lo, d, out_shift);
                    hn::Store(out01_i_lo, d, out_shift + lanes);
                    hn::Store(out23_r_lo, d, out_shift + lanes * 2);
                    hn::Store(out23_i_lo, d, out_shift + lanes * 3);
                    hn::Store(out01_r_hi, d, out_shift + lanes * 4);
                    hn::Store(out01_i_hi, d, out_shift + lanes * 5);
                    hn::Store(out23_r_hi, d, out_shift + lanes * 6);
                    hn::Store(out23_i_hi, d, out_shift + lanes * 7);
                } else {
                    const size_t k = (v * lanes) & 3;
                    F* HWY_RESTRICT out_shift = out_aosoa + (i << 3) + k * 2;

                    hn::Store(out0_r, d, out_shift);
                    hn::Store(out0_i, d, out_shift + lanes);
                    hn::Store(out1_r, d, out_shift + 8);
                    hn::Store(out1_i, d, out_shift + 8 + lanes);
                    hn::Store(out2_r, d, out_shift + 16);
                    hn::Store(out2_i, d, out_shift + 16 + lanes);
                    hn::Store(out3_r, d, out_shift + 24);
                    hn::Store(out3_i, d, out_shift + 24 + lanes);
                }
            }
        }
    }

    /**
     * perform a Stockham DIT radix-4 pass and convert data from AoSoA to AoS/SoA
     * @tparam is_forward
     * @tparam F
     * @tparam Ptr
     * @param in_aosoa
     * @param out
     * @param n
     * @param width
     * @param w_ptr
     */
    template <bool is_forward, typename F, typename Ptr>
    inline void radix4_last_pass_fused_aosoa(const F* HWY_RESTRICT in_aosoa,
                                             Ptr out,
                                             const size_t n, const size_t width, const F* HWY_RESTRICT w_ptr) {
        in_aosoa = static_cast<const F*>(HWY_ASSUME_ALIGNED(in_aosoa, HWY_ALIGNMENT));
        w_ptr = static_cast<const F*>(HWY_ASSUME_ALIGNED(w_ptr, HWY_ALIGNMENT));

        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        const size_t quarter_n = n >> 2;
        const size_t half_n = n >> 1;
        const size_t three_quarter_n = quarter_n * 3;

        const size_t out_offset1 = Ptr::get_complex_offset(width);
        const size_t out_offset2 = Ptr::get_complex_offset(width << 1);
        const size_t out_offset3 = out_offset1 + out_offset2;

        const size_t mask = width - 1;

        HWY_ASSUME(quarter_n >= lanes);
        HWY_ASSUME((quarter_n % lanes) == 0);
        for (size_t i = 0; i < quarter_n; i += lanes) {
            const size_t k = i & mask;
            const F* HWY_RESTRICT w_shift = w_ptr + k * 6;

            hn::Vec<decltype(d)> t1_r, t1_i;
            {
                const auto w1_r = hn::Load(d, w_shift);
                const auto w1_i = hn::Load(d, w_shift + lanes);
                const auto r1 = hn::Load(d, in_aosoa + 2 * (quarter_n + i));
                const auto i1 = hn::Load(d, in_aosoa + 2 * (quarter_n + i) + lanes);
                t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
                t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));
            }
            hn::Vec<decltype(d)> t3_r, t3_i;
            {
                const auto w3_r = hn::Load(d, w_shift + lanes * 4);
                const auto w3_i = hn::Load(d, w_shift + lanes * 5);
                const auto r3 = hn::Load(d, in_aosoa + 2 * (three_quarter_n + i));
                const auto i3 = hn::Load(d, in_aosoa + 2 * (three_quarter_n + i) + lanes);
                t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
                t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));
            }

            const auto s2_r = hn::Add(t1_r, t3_r);
            const auto s2_i = hn::Add(t1_i, t3_i);
            const auto s3_r = hn::Sub(t1_r, t3_r);
            const auto s3_i = hn::Sub(t1_i, t3_i);

            hn::Vec<decltype(d)> t2_r, t2_i;
            {
                const auto w2_r = hn::Load(d, w_shift + lanes * 2);
                const auto w2_i = hn::Load(d, w_shift + lanes * 3);
                const auto r2 = hn::Load(d, in_aosoa + 2 * (half_n + i));
                const auto i2 = hn::Load(d, in_aosoa + 2 * (half_n + i) + lanes);
                t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
                t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));
            }

            const auto r0 = hn::Load(d, in_aosoa + 2 * i);
            const auto i0 = hn::Load(d, in_aosoa + 2 * i + lanes);
            const auto s0_r = hn::Add(r0, t2_r);
            const auto s0_i = hn::Add(i0, t2_i);
            const auto s1_r = hn::Sub(r0, t2_r);
            const auto s1_i = hn::Sub(i0, t2_i);

            const size_t j_times_4 = (i & ~mask) << 2;
            const size_t out_idx = j_times_4 + k;
            Ptr out_shift = out.shift(Ptr::get_complex_offset(out_idx));

            store_complex<is_forward>(d, out_shift, hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i));
            store_complex<is_forward>(d, out_shift.shift(out_offset2), hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i));
            store_complex<is_forward>(d, out_shift.shift(out_offset1), hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r));
            store_complex<is_forward>(d, out_shift.shift(out_offset3), hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r));
        }
    }

    /**
     * performs a Stockham DIT radix-8 first pass and convert data from AoS/SoA to AoSoA
     * @tparam is_forward
     * @tparam F
     * @tparam Ptr
     * @param in
     * @param out_aosoa
     * @param n
     */
    template <bool is_forward, typename F, typename Ptr>
    inline void radix8_first_pass_fused_aosoa(Ptr in,
                                              F* HWY_RESTRICT out_aosoa,
                                              const size_t n) {
        out_aosoa = static_cast<F*>(HWY_ASSUME_ALIGNED(out_aosoa, HWY_ALIGNMENT));

        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        const size_t one_eight_n = n >> 3;

        const size_t in_offset1 = Ptr::get_complex_offset(one_eight_n);
        const size_t in_offset2 = Ptr::get_complex_offset(n >> 2);
        const size_t in_offset3 = in_offset1 + in_offset2;
        const size_t in_offset4 = Ptr::get_complex_offset(n >> 1);
        const size_t in_offset5 = in_offset1 + in_offset4;
        const size_t in_offset6 = in_offset2 + in_offset4;
        const size_t in_offset7 = in_offset1 + in_offset6;

        static constexpr F kInvSqrt2 = static_cast<F>(1.0 / std::numbers::sqrt2);
        const auto inv_sqrt2 = hn::Set(d, kInvSqrt2);

        HWY_ASSUME(one_eight_n >= lanes);
        HWY_ASSUME((one_eight_n % lanes) == 0);
        for (size_t j = 0; j + lanes <= one_eight_n; j += lanes) {
            const Ptr in_shift = in.shift(Ptr::get_complex_offset(j));

            hn::Vec<decltype(d)> y00_r, y00_i, y02_r, y02_i, y01_r, y01_i, y03_r, y03_i;
            {
                hn::Vec<decltype(d)> a_r, a_i, b_r, b_i;

                load_complex<is_forward>(d, in_shift, a_r, a_i);
                load_complex<is_forward>(d, in_shift.shift(in_offset4), b_r, b_i);
                const auto t0_r = hn::Add(a_r, b_r), t0_i = hn::Add(a_i, b_i);
                const auto t1_r = hn::Sub(a_r, b_r), t1_i = hn::Sub(a_i, b_i);

                load_complex<is_forward>(d, in_shift.shift(in_offset2), a_r, a_i);
                load_complex<is_forward>(d, in_shift.shift(in_offset6), b_r, b_i);
                const auto t2_r = hn::Add(a_r, b_r), t2_i = hn::Add(a_i, b_i);
                const auto t3_r = hn::Sub(a_r, b_r), t3_i = hn::Sub(a_i, b_i);

                y00_r = hn::Add(t0_r, t2_r);
                y00_i = hn::Add(t0_i, t2_i);
                y02_r = hn::Sub(t0_r, t2_r);
                y02_i = hn::Sub(t0_i, t2_i);
                y01_r = hn::Add(t1_r, t3_i);
                y01_i = hn::Sub(t1_i, t3_r);
                y03_r = hn::Sub(t1_r, t3_i);
                y03_i = hn::Add(t1_i, t3_r);
            }

            hn::Vec<decltype(d)> y10_r, y10_i, y12_r, y12_i, y11_r, y11_i, y13_r, y13_i;
            {
                hn::Vec<decltype(d)> a_r, a_i, b_r, b_i;

                load_complex<is_forward>(d, in_shift.shift(in_offset1), a_r, a_i);
                load_complex<is_forward>(d, in_shift.shift(in_offset5), b_r, b_i);
                const auto u0_r = hn::Add(a_r, b_r), u0_i = hn::Add(a_i, b_i);
                const auto u1_r = hn::Sub(a_r, b_r), u1_i = hn::Sub(a_i, b_i);

                load_complex<is_forward>(d, in_shift.shift(in_offset3), a_r, a_i);
                load_complex<is_forward>(d, in_shift.shift(in_offset7), b_r, b_i);
                const auto u2_r = hn::Add(a_r, b_r), u2_i = hn::Add(a_i, b_i);
                const auto u3_r = hn::Sub(a_r, b_r), u3_i = hn::Sub(a_i, b_i);

                y10_r = hn::Add(u0_r, u2_r);
                y10_i = hn::Add(u0_i, u2_i);
                y12_r = hn::Sub(u0_r, u2_r);
                y12_i = hn::Sub(u0_i, u2_i);
                y11_r = hn::Add(u1_r, u3_i);
                y11_i = hn::Sub(u1_i, u3_r);
                y13_r = hn::Sub(u1_r, u3_i);
                y13_i = hn::Add(u1_i, u3_r);
            }

            {
                const auto v1_r = hn::Mul(hn::Add(y11_r, y11_i), inv_sqrt2);
                const auto v1_i = hn::Mul(hn::Sub(y11_i, y11_r), inv_sqrt2);
                y11_r = v1_r;
                y11_i = v1_i;

                const auto v2_r = y12_i;
                const auto v2_i = hn::Neg(y12_r);
                y12_r = v2_r;
                y12_i = v2_i;

                const auto v3_r = hn::Mul(hn::Sub(y13_i, y13_r), inv_sqrt2);
                const auto v3_i = hn::Mul(hn::Neg(hn::Add(y13_r, y13_i)), inv_sqrt2);
                y13_r = v3_r;
                y13_i = v3_i;
            }

            F* HWY_RESTRICT out_shift = out_aosoa + (j << 4);
            static constexpr bool is_wide = (lanes > 4) || (sizeof(F) == 8 && lanes == 4);
            {
                const auto il_00 = hn::InterleaveLower(d, hn::Add(y00_r, y10_r), hn::Sub(y00_r, y10_r));
                const auto il_01 = hn::InterleaveLower(d, hn::Add(y01_r, y11_r), hn::Sub(y01_r, y11_r));
                const auto il_02 = hn::InterleaveLower(d, hn::Add(y02_r, y12_r), hn::Sub(y02_r, y12_r));
                const auto il_03 = hn::InterleaveLower(d, hn::Add(y03_r, y13_r), hn::Sub(y03_r, y13_r));

                hn::Vec<decltype(d)> r0, r1, r2, r3;
                transpose4x4(d, il_00, il_01, il_02, il_03, r0, r1, r2, r3);

                if constexpr (is_wide) {
                    hn::Store(r0, d, out_shift);
                    hn::Store(r1, d, out_shift + 2 * lanes);
                    hn::Store(r2, d, out_shift + 8 * lanes);
                    hn::Store(r3, d, out_shift + 10 * lanes);
                } else {
                    hn::Store(r0, d, out_shift);
                    hn::Store(r1, d, out_shift + 2 * lanes);
                    hn::Store(r2, d, out_shift + 4 * lanes);
                    hn::Store(r3, d, out_shift + 6 * lanes);
                }
            }

            {
                const auto iu_00 = hn::InterleaveUpper(d, hn::Add(y00_r, y10_r), hn::Sub(y00_r, y10_r));
                const auto iu_01 = hn::InterleaveUpper(d, hn::Add(y01_r, y11_r), hn::Sub(y01_r, y11_r));
                const auto iu_02 = hn::InterleaveUpper(d, hn::Add(y02_r, y12_r), hn::Sub(y02_r, y12_r));
                const auto iu_03 = hn::InterleaveUpper(d, hn::Add(y03_r, y13_r), hn::Sub(y03_r, y13_r));

                hn::Vec<decltype(d)> r0, r1, r2, r3;
                transpose4x4(d, iu_00, iu_01, iu_02, iu_03, r0, r1, r2, r3);

                if constexpr (is_wide) {
                    hn::Store(r0, d, out_shift + 4 * lanes);
                    hn::Store(r1, d, out_shift + 6 * lanes);
                    hn::Store(r2, d, out_shift + 12 * lanes);
                    hn::Store(r3, d, out_shift + 14 * lanes);
                } else {
                    hn::Store(r0, d, out_shift + 8 * lanes);
                    hn::Store(r1, d, out_shift + 10 * lanes);
                    hn::Store(r2, d, out_shift + 12 * lanes);
                    hn::Store(r3, d, out_shift + 14 * lanes);
                }
            }

            {
                const auto il_00 = hn::InterleaveLower(d, hn::Add(y00_i, y10_i), hn::Sub(y00_i, y10_i));
                const auto il_01 = hn::InterleaveLower(d, hn::Add(y01_i, y11_i), hn::Sub(y01_i, y11_i));
                const auto il_02 = hn::InterleaveLower(d, hn::Add(y02_i, y12_i), hn::Sub(y02_i, y12_i));
                const auto il_03 = hn::InterleaveLower(d, hn::Add(y03_i, y13_i), hn::Sub(y03_i, y13_i));

                hn::Vec<decltype(d)> i0, i1, i2, i3;
                transpose4x4(d, il_00, il_01, il_02, il_03, i0, i1, i2, i3);

                if constexpr (is_wide) {
                    hn::Store(i0, d, out_shift + lanes);
                    hn::Store(i1, d, out_shift + 3 * lanes);
                    hn::Store(i2, d, out_shift + 9 * lanes);
                    hn::Store(i3, d, out_shift + 11 * lanes);
                } else {
                    hn::Store(i0, d, out_shift + lanes);
                    hn::Store(i1, d, out_shift + 3 * lanes);
                    hn::Store(i2, d, out_shift + 5 * lanes);
                    hn::Store(i3, d, out_shift + 7 * lanes);
                }
            }

            {
                const auto iu_00 = hn::InterleaveUpper(d, hn::Add(y00_i, y10_i), hn::Sub(y00_i, y10_i));
                const auto iu_01 = hn::InterleaveUpper(d, hn::Add(y01_i, y11_i), hn::Sub(y01_i, y11_i));
                const auto iu_02 = hn::InterleaveUpper(d, hn::Add(y02_i, y12_i), hn::Sub(y02_i, y12_i));
                const auto iu_03 = hn::InterleaveUpper(d, hn::Add(y03_i, y13_i), hn::Sub(y03_i, y13_i));

                hn::Vec<decltype(d)> i0, i1, i2, i3;
                transpose4x4(d, iu_00, iu_01, iu_02, iu_03, i0, i1, i2, i3);

                if constexpr (is_wide) {
                    hn::Store(i0, d, out_shift + 5 * lanes);
                    hn::Store(i1, d, out_shift + 7 * lanes);
                    hn::Store(i2, d, out_shift + 13 * lanes);
                    hn::Store(i3, d, out_shift + 15 * lanes);
                } else {
                    hn::Store(i0, d, out_shift + 9 * lanes);
                    hn::Store(i1, d, out_shift + 11 * lanes);
                    hn::Store(i2, d, out_shift + 13 * lanes);
                    hn::Store(i3, d, out_shift + 15 * lanes);
                }
            }
        }
    }

    /**
     * performs a Cooley-Tukey DIF radix-4 first pass and converts data from AoS/SoA to AoSoA
     * @tparam is_forward
     * @tparam F
     * @tparam Ptr
     * @param in
     * @param out_aosoa
     * @param n
     * @param w_ptr
     */
    template <bool is_forward, typename F, typename Ptr>
    inline void radix4_first_pass_dif_fused_aosoa(Ptr in,
                                                  F* HWY_RESTRICT out_aosoa,
                                                  const size_t n,
                                                  const F* HWY_RESTRICT w_ptr) {
        out_aosoa = static_cast<F*>(HWY_ASSUME_ALIGNED(out_aosoa, HWY_ALIGNMENT));

        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        const size_t quarter_n = n >> 2;

        const size_t in_offset1 = Ptr::get_complex_offset(quarter_n);
        const size_t in_offset2 = Ptr::get_complex_offset(n >> 1);
        const size_t in_offset3 = in_offset1 + in_offset2;

        const size_t out_offset1 = quarter_n << 1;
        const size_t out_offset2 = n;
        const size_t out_offset3 = out_offset1 + out_offset2;

        HWY_ASSUME(quarter_n >= lanes);
        HWY_ASSUME((quarter_n % lanes) == 0);
        for (size_t i = 0; i < quarter_n; i += lanes) {
            const Ptr in_shift = in.shift(Ptr::get_complex_offset(i));

            const F* HWY_RESTRICT w_shift = w_ptr + i * 6;
            F* HWY_RESTRICT out_shift = out_aosoa + (i << 1);

            hn::Vec<decltype(d)> r0, i0, r2, i2;
            load_complex<is_forward>(d, in_shift, r0, i0);
            load_complex<is_forward>(d, in_shift.shift(in_offset2), r2, i2);
            const auto t0_r = hn::Add(r0, r2);
            const auto t0_i = hn::Add(i0, i2);
            const auto t1_r = hn::Sub(r0, r2);
            const auto t1_i = hn::Sub(i0, i2);

            hn::Vec<decltype(d)> r1, i1, r3, i3;
            load_complex<is_forward>(d, in_shift.shift(in_offset1), r1, i1);
            load_complex<is_forward>(d, in_shift.shift(in_offset3), r3, i3);
            const auto t2_r = hn::Add(r1, r3);
            const auto t2_i = hn::Add(i1, i3);
            const auto t3_r = hn::Sub(r1, r3);
            const auto t3_i = hn::Sub(i1, i3);
            {
                const auto y0_r = hn::Add(t0_r, t2_r);
                const auto y0_i = hn::Add(t0_i, t2_i);
                hn::Store(y0_r, d, out_shift);
                hn::Store(y0_i, d, out_shift + lanes);
            }
            {
                const auto y2_r = hn::Sub(t0_r, t2_r);
                const auto y2_i = hn::Sub(t0_i, t2_i);
                const auto w2_r = hn::Load(d, w_shift + lanes * 2);
                const auto w2_i = hn::Load(d, w_shift + lanes * 3);
                const auto out2_r = hn::NegMulAdd(y2_i, w2_i, hn::Mul(y2_r, w2_r));
                const auto out2_i = hn::MulAdd(y2_i, w2_r, hn::Mul(y2_r, w2_i));

                hn::Store(out2_r, d, out_shift + out_offset2);
                hn::Store(out2_i, d, out_shift + out_offset2 + lanes);
            }
            {
                const auto y1_r = hn::Add(t1_r, t3_i);
                const auto y1_i = hn::Sub(t1_i, t3_r);
                const auto w1_r = hn::Load(d, w_shift);
                const auto w1_i = hn::Load(d, w_shift + lanes);
                const auto out1_r = hn::NegMulAdd(y1_i, w1_i, hn::Mul(y1_r, w1_r));
                const auto out1_i = hn::MulAdd(y1_i, w1_r, hn::Mul(y1_r, w1_i));

                hn::Store(out1_r, d, out_shift + out_offset1);
                hn::Store(out1_i, d, out_shift + out_offset1 + lanes);
            }
            {
                const auto y3_r = hn::Sub(t1_r, t3_i);
                const auto y3_i = hn::Add(t1_i, t3_r);
                const auto w3_r = hn::Load(d, w_shift + lanes * 4);
                const auto w3_i = hn::Load(d, w_shift + lanes * 5);
                const auto out3_r = hn::NegMulAdd(y3_i, w3_i, hn::Mul(y3_r, w3_r));
                const auto out3_i = hn::MulAdd(y3_i, w3_r, hn::Mul(y3_r, w3_i));

                hn::Store(out3_r, d, out_shift + out_offset3);
                hn::Store(out3_i, d, out_shift + out_offset3 + lanes);
            }
        }
    }

    /**
     * perform a Cooley-Tukey DIF radix-4 pass when width >= 8
     * @tparam F
     * @param workspace
     * @param n
     * @param width
     * @param w_ptr
     */
    template <typename F>
    inline void radix4_dif_aosoa_inplace(F* HWY_RESTRICT workspace,
                                         const size_t n, const size_t width, const F* HWY_RESTRICT w_ptr) {
        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        const size_t sub_n = width << 2;

        const size_t offset1 = width << 1;
        const size_t offset2 = width << 2;
        const size_t offset3 = offset1 + offset2;

        for (size_t block = 0; block < n; block += sub_n) {
            F* HWY_RESTRICT ptr_block = workspace + (block << 1);
            HWY_ASSUME(width >= lanes);
            HWY_ASSUME((width % lanes) == 0);
            for (size_t i = 0; i < width; i += lanes) {
                const F* HWY_RESTRICT w_shift = w_ptr + i * 6;
                F* HWY_RESTRICT ptr_shift = ptr_block + (i << 1);

                const auto r0 = hn::Load(d, ptr_shift);
                const auto i0 = hn::Load(d, ptr_shift + lanes);
                const auto r2 = hn::Load(d, ptr_shift + offset2);
                const auto i2 = hn::Load(d, ptr_shift + offset2 + lanes);

                const auto t0_r = hn::Add(r0, r2);
                const auto t0_i = hn::Add(i0, i2);
                const auto t1_r = hn::Sub(r0, r2);
                const auto t1_i = hn::Sub(i0, i2);

                const auto r1 = hn::Load(d, ptr_shift + offset1);
                const auto i1 = hn::Load(d, ptr_shift + offset1 + lanes);
                const auto r3 = hn::Load(d, ptr_shift + offset3);
                const auto i3 = hn::Load(d, ptr_shift + offset3 + lanes);

                const auto t2_r = hn::Add(r1, r3);
                const auto t2_i = hn::Add(i1, i3);
                const auto t3_r = hn::Sub(r1, r3);
                const auto t3_i = hn::Sub(i1, i3);

                {
                    const auto y0_r = hn::Add(t0_r, t2_r);
                    const auto y0_i = hn::Add(t0_i, t2_i);

                    hn::Store(y0_r, d, ptr_shift);
                    hn::Store(y0_i, d, ptr_shift + lanes);
                }
                {
                    const auto y2_r = hn::Sub(t0_r, t2_r);
                    const auto y2_i = hn::Sub(t0_i, t2_i);
                    const auto w2_r = hn::Load(d, w_shift + lanes * 2);
                    const auto w2_i = hn::Load(d, w_shift + lanes * 3);
                    const auto out2_r = hn::NegMulAdd(y2_i, w2_i, hn::Mul(y2_r, w2_r));
                    const auto out2_i = hn::MulAdd(y2_i, w2_r, hn::Mul(y2_r, w2_i));

                    hn::Store(out2_r, d, ptr_shift + offset2);
                    hn::Store(out2_i, d, ptr_shift + offset2 + lanes);
                }
                {
                    const auto y1_r = hn::Add(t1_r, t3_i);
                    const auto y1_i = hn::Sub(t1_i, t3_r);
                    const auto w1_r = hn::Load(d, w_shift);
                    const auto w1_i = hn::Load(d, w_shift + lanes);
                    const auto out1_r = hn::NegMulAdd(y1_i, w1_i, hn::Mul(y1_r, w1_r));
                    const auto out1_i = hn::MulAdd(y1_i, w1_r, hn::Mul(y1_r, w1_i));

                    hn::Store(out1_r, d, ptr_shift + offset1);
                    hn::Store(out1_i, d, ptr_shift + offset1 + lanes);
                }
                {
                    const auto y3_r = hn::Sub(t1_r, t3_i);
                    const auto y3_i = hn::Add(t1_i, t3_r);
                    const auto w3_r = hn::Load(d, w_shift + lanes * 4);
                    const auto w3_i = hn::Load(d, w_shift + lanes * 5);
                    const auto out3_r = hn::NegMulAdd(y3_i, w3_i, hn::Mul(y3_r, w3_r));
                    const auto out3_i = hn::MulAdd(y3_i, w3_r, hn::Mul(y3_r, w3_i));

                    hn::Store(out3_r, d, ptr_shift + offset3);
                    hn::Store(out3_i, d, ptr_shift + offset3 + lanes);
                }
            }
        }
    }

    /**
     * performs a forward Stockham DIT radix-4 first pass on AoSoA
     * @tparam F
     * @param in_aosoa
     * @param out_aosoa
     * @param n
     */
    template <typename F>
    inline void radix4_first_pass_aosoa(const F* HWY_RESTRICT in_aosoa,
                                        F* HWY_RESTRICT out_aosoa,
                                        const size_t n) {
        in_aosoa = static_cast<const F*>(HWY_ASSUME_ALIGNED(in_aosoa, HWY_ALIGNMENT));
        out_aosoa = static_cast<F*>(HWY_ASSUME_ALIGNED(out_aosoa, HWY_ALIGNMENT));

        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        const size_t quarter_n = n >> 2;

        const size_t in_offset1 = n >> 1;
        const size_t in_offset2 = n;
        const size_t in_offset3 = in_offset1 + in_offset2;

        HWY_ASSUME(quarter_n >= lanes);
        HWY_ASSUME((quarter_n % lanes) == 0);

        for (size_t j = 0; j < quarter_n; j += lanes) {
            const F* HWY_RESTRICT in_shift = in_aosoa + (j << 1);

            const auto x0_r = hn::Load(d, in_shift);
            const auto x0_i = hn::Load(d, in_shift + lanes);
            const auto x2_r = hn::Load(d, in_shift + in_offset2);
            const auto x2_i = hn::Load(d, in_shift + in_offset2 + lanes);

            const auto t0_r = hn::Add(x0_r, x2_r);
            const auto t0_i = hn::Add(x0_i, x2_i);
            const auto t1_r = hn::Sub(x0_r, x2_r);
            const auto t1_i = hn::Sub(x0_i, x2_i);

            const auto x1_r = hn::Load(d, in_shift + in_offset1);
            const auto x1_i = hn::Load(d, in_shift + in_offset1 + lanes);
            const auto x3_r = hn::Load(d, in_shift + in_offset3);
            const auto x3_i = hn::Load(d, in_shift + in_offset3 + lanes);

            const auto t2_r = hn::Add(x1_r, x3_r);
            const auto t2_i = hn::Add(x1_i, x3_i);
            const auto t3_r = hn::Sub(x1_r, x3_r);
            const auto t3_i = hn::Sub(x1_i, x3_i);

            F* HWY_RESTRICT out_shift = out_aosoa + (j << 3);
            {
                const auto out0_r = hn::Add(t0_r, t2_r);
                const auto out2_r = hn::Sub(t0_r, t2_r);
                const auto out1_r = hn::Add(t1_r, t3_i);
                const auto out3_r = hn::Sub(t1_r, t3_i);

                hn::Vec<decltype(d)> r0, r1, r2, r3;
                transpose4x4(d, out0_r, out1_r, out2_r, out3_r, r0, r1, r2, r3);

                hn::Store(r0, d, out_shift);
                hn::Store(r1, d, out_shift + 2 * lanes);
                hn::Store(r2, d, out_shift + 4 * lanes);
                hn::Store(r3, d, out_shift + 6 * lanes);
            }
            {
                const auto out0_i = hn::Add(t0_i, t2_i);
                const auto out2_i = hn::Sub(t0_i, t2_i);
                const auto out1_i = hn::Sub(t1_i, t3_r);
                const auto out3_i = hn::Add(t1_i, t3_r);

                hn::Vec<decltype(d)> i0, i1, i2, i3;
                transpose4x4(d, out0_i, out1_i, out2_i, out3_i, i0, i1, i2, i3);

                hn::Store(i0, d, out_shift + lanes);
                hn::Store(i1, d, out_shift + 3 * lanes);
                hn::Store(i2, d, out_shift + 5 * lanes);
                hn::Store(i3, d, out_shift + 7 * lanes);
            }
        }
    }

    /**
     * performs a forward Stockham DIT radix-8 first pass on AoSoA
     * @tparam F
     * @param in_aosoa
     * @param out_aosoa
     * @param n
     */
    template <typename F>
    inline void radix8_first_pass_aosoa(F* HWY_RESTRICT in_aosoa,
                                        F* HWY_RESTRICT out_aosoa,
                                        const size_t n) {
        out_aosoa = static_cast<F*>(HWY_ASSUME_ALIGNED(out_aosoa, HWY_ALIGNMENT));

        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        const size_t one_eight_n = n >> 3;

        const size_t in_offset1 = n >> 2;
        const size_t in_offset2 = n >> 1;
        const size_t in_offset3 = in_offset1 + in_offset2;
        const size_t in_offset4 = n;
        const size_t in_offset5 = in_offset1 + in_offset4;
        const size_t in_offset6 = in_offset2 + in_offset4;
        const size_t in_offset7 = in_offset1 + in_offset6;

        static constexpr F kInvSqrt2 = static_cast<F>(1.0 / std::numbers::sqrt2);
        const auto inv_sqrt2 = hn::Set(d, kInvSqrt2);

        HWY_ASSUME(one_eight_n >= lanes);
        HWY_ASSUME((one_eight_n % lanes) == 0);
        for (size_t j = 0; j + lanes <= one_eight_n; j += lanes) {
            const F* HWY_RESTRICT in_shift = in_aosoa + (j << 1);

            hn::Vec<decltype(d)> y00_r, y00_i, y02_r, y02_i, y01_r, y01_i, y03_r, y03_i;
            {
                hn::Vec<decltype(d)> a_r, a_i, b_r, b_i;

                a_r = hn::Load(d, in_shift);
                a_i = hn::Load(d, in_shift + lanes);
                b_r = hn::Load(d, in_shift + in_offset4);
                b_i = hn::Load(d, in_shift + in_offset4 + lanes);
                const auto t0_r = hn::Add(a_r, b_r), t0_i = hn::Add(a_i, b_i);
                const auto t1_r = hn::Sub(a_r, b_r), t1_i = hn::Sub(a_i, b_i);

                a_r = hn::Load(d, in_shift + in_offset2);
                a_i = hn::Load(d, in_shift + in_offset2 + lanes);
                b_r = hn::Load(d, in_shift + in_offset6);
                b_i = hn::Load(d, in_shift + in_offset6 + lanes);
                const auto t2_r = hn::Add(a_r, b_r), t2_i = hn::Add(a_i, b_i);
                const auto t3_r = hn::Sub(a_r, b_r), t3_i = hn::Sub(a_i, b_i);

                y00_r = hn::Add(t0_r, t2_r);
                y00_i = hn::Add(t0_i, t2_i);
                y02_r = hn::Sub(t0_r, t2_r);
                y02_i = hn::Sub(t0_i, t2_i);
                y01_r = hn::Add(t1_r, t3_i);
                y01_i = hn::Sub(t1_i, t3_r);
                y03_r = hn::Sub(t1_r, t3_i);
                y03_i = hn::Add(t1_i, t3_r);
            }

            hn::Vec<decltype(d)> y10_r, y10_i, y12_r, y12_i, y11_r, y11_i, y13_r, y13_i;
            {
                hn::Vec<decltype(d)> a_r, a_i, b_r, b_i;

                a_r = hn::Load(d, in_shift + in_offset1);
                a_i = hn::Load(d, in_shift + in_offset1 + lanes);
                b_r = hn::Load(d, in_shift + in_offset5);
                b_i = hn::Load(d, in_shift + in_offset5 + lanes);
                const auto u0_r = hn::Add(a_r, b_r), u0_i = hn::Add(a_i, b_i);
                const auto u1_r = hn::Sub(a_r, b_r), u1_i = hn::Sub(a_i, b_i);

                a_r = hn::Load(d, in_shift + in_offset3);
                a_i = hn::Load(d, in_shift + in_offset3 + lanes);
                b_r = hn::Load(d, in_shift + in_offset7);
                b_i = hn::Load(d, in_shift + in_offset7 + lanes);
                const auto u2_r = hn::Add(a_r, b_r), u2_i = hn::Add(a_i, b_i);
                const auto u3_r = hn::Sub(a_r, b_r), u3_i = hn::Sub(a_i, b_i);

                y10_r = hn::Add(u0_r, u2_r);
                y10_i = hn::Add(u0_i, u2_i);
                y12_r = hn::Sub(u0_r, u2_r);
                y12_i = hn::Sub(u0_i, u2_i);
                y11_r = hn::Add(u1_r, u3_i);
                y11_i = hn::Sub(u1_i, u3_r);
                y13_r = hn::Sub(u1_r, u3_i);
                y13_i = hn::Add(u1_i, u3_r);
            }
            {
                const auto v1_r = hn::Mul(hn::Add(y11_r, y11_i), inv_sqrt2);
                const auto v1_i = hn::Mul(hn::Sub(y11_i, y11_r), inv_sqrt2);
                y11_r = v1_r;
                y11_i = v1_i;

                const auto v2_r = y12_i;
                const auto v2_i = hn::Neg(y12_r);
                y12_r = v2_r;
                y12_i = v2_i;

                const auto v3_r = hn::Mul(hn::Sub(y13_i, y13_r), inv_sqrt2);
                const auto v3_i = hn::Mul(hn::Neg(hn::Add(y13_r, y13_i)), inv_sqrt2);
                y13_r = v3_r;
                y13_i = v3_i;
            }
            F* HWY_RESTRICT out_shift = out_aosoa + (j << 4);
            static constexpr bool is_wide = (lanes > 4) || (sizeof(F) == 8 && lanes == 4);
            {
                const auto il_00 = hn::InterleaveLower(d, hn::Add(y00_r, y10_r), hn::Sub(y00_r, y10_r));
                const auto il_01 = hn::InterleaveLower(d, hn::Add(y01_r, y11_r), hn::Sub(y01_r, y11_r));
                const auto il_02 = hn::InterleaveLower(d, hn::Add(y02_r, y12_r), hn::Sub(y02_r, y12_r));
                const auto il_03 = hn::InterleaveLower(d, hn::Add(y03_r, y13_r), hn::Sub(y03_r, y13_r));

                hn::Vec<decltype(d)> r0, r1, r2, r3;
                transpose4x4(d, il_00, il_01, il_02, il_03, r0, r1, r2, r3);

                if constexpr (is_wide) {
                    hn::Store(r0, d, out_shift);
                    hn::Store(r1, d, out_shift + 2 * lanes);
                    hn::Store(r2, d, out_shift + 8 * lanes);
                    hn::Store(r3, d, out_shift + 10 * lanes);
                } else {
                    hn::Store(r0, d, out_shift);
                    hn::Store(r1, d, out_shift + 2 * lanes);
                    hn::Store(r2, d, out_shift + 4 * lanes);
                    hn::Store(r3, d, out_shift + 6 * lanes);
                }
            }
            {
                const auto iu_00 = hn::InterleaveUpper(d, hn::Add(y00_r, y10_r), hn::Sub(y00_r, y10_r));
                const auto iu_01 = hn::InterleaveUpper(d, hn::Add(y01_r, y11_r), hn::Sub(y01_r, y11_r));
                const auto iu_02 = hn::InterleaveUpper(d, hn::Add(y02_r, y12_r), hn::Sub(y02_r, y12_r));
                const auto iu_03 = hn::InterleaveUpper(d, hn::Add(y03_r, y13_r), hn::Sub(y03_r, y13_r));

                hn::Vec<decltype(d)> r0, r1, r2, r3;
                transpose4x4(d, iu_00, iu_01, iu_02, iu_03, r0, r1, r2, r3);

                if constexpr (is_wide) {
                    hn::Store(r0, d, out_shift + 4 * lanes);
                    hn::Store(r1, d, out_shift + 6 * lanes);
                    hn::Store(r2, d, out_shift + 12 * lanes);
                    hn::Store(r3, d, out_shift + 14 * lanes);
                } else {
                    hn::Store(r0, d, out_shift + 8 * lanes);
                    hn::Store(r1, d, out_shift + 10 * lanes);
                    hn::Store(r2, d, out_shift + 12 * lanes);
                    hn::Store(r3, d, out_shift + 14 * lanes);
                }
            }
            {
                const auto il_00 = hn::InterleaveLower(d, hn::Add(y00_i, y10_i), hn::Sub(y00_i, y10_i));
                const auto il_01 = hn::InterleaveLower(d, hn::Add(y01_i, y11_i), hn::Sub(y01_i, y11_i));
                const auto il_02 = hn::InterleaveLower(d, hn::Add(y02_i, y12_i), hn::Sub(y02_i, y12_i));
                const auto il_03 = hn::InterleaveLower(d, hn::Add(y03_i, y13_i), hn::Sub(y03_i, y13_i));

                hn::Vec<decltype(d)> i0, i1, i2, i3;
                transpose4x4(d, il_00, il_01, il_02, il_03, i0, i1, i2, i3);

                if constexpr (is_wide) {
                    hn::Store(i0, d, out_shift + lanes);
                    hn::Store(i1, d, out_shift + 3 * lanes);
                    hn::Store(i2, d, out_shift + 9 * lanes);
                    hn::Store(i3, d, out_shift + 11 * lanes);
                } else {
                    hn::Store(i0, d, out_shift + lanes);
                    hn::Store(i1, d, out_shift + 3 * lanes);
                    hn::Store(i2, d, out_shift + 5 * lanes);
                    hn::Store(i3, d, out_shift + 7 * lanes);
                }
            }
            {
                const auto iu_00 = hn::InterleaveUpper(d, hn::Add(y00_i, y10_i), hn::Sub(y00_i, y10_i));
                const auto iu_01 = hn::InterleaveUpper(d, hn::Add(y01_i, y11_i), hn::Sub(y01_i, y11_i));
                const auto iu_02 = hn::InterleaveUpper(d, hn::Add(y02_i, y12_i), hn::Sub(y02_i, y12_i));
                const auto iu_03 = hn::InterleaveUpper(d, hn::Add(y03_i, y13_i), hn::Sub(y03_i, y13_i));

                hn::Vec<decltype(d)> i0, i1, i2, i3;
                transpose4x4(d, iu_00, iu_01, iu_02, iu_03, i0, i1, i2, i3);

                if constexpr (is_wide) {
                    hn::Store(i0, d, out_shift + 5 * lanes);
                    hn::Store(i1, d, out_shift + 7 * lanes);
                    hn::Store(i2, d, out_shift + 13 * lanes);
                    hn::Store(i3, d, out_shift + 15 * lanes);
                } else {
                    hn::Store(i0, d, out_shift + 9 * lanes);
                    hn::Store(i1, d, out_shift + 11 * lanes);
                    hn::Store(i2, d, out_shift + 13 * lanes);
                    hn::Store(i3, d, out_shift + 15 * lanes);
                }
            }
        }
    }
}
