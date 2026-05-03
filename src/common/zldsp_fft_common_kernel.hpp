#pragma once

#include "zldsp_fft_common_init.hpp"
#include "zldsp_fft_common_structure.hpp"

namespace zldsp::fft::common {
    namespace hn = hwy::HWY_NAMESPACE;

    /**
     * hardcoded FFT for order = 0
     * @tparam is_forward
     * @tparam F
     * @tparam InPtr
     * @tparam OutPtr
     * @param in
     * @param out
     */
    template <bool is_forward, typename F, typename InPtr, typename OutPtr>
    inline void callback_order_0(InPtr in, OutPtr out) {
        F r, i;
        load_scalar<is_forward>(in, r, i);
        store_scalar<is_forward>(out, r, i);
    }

    /**
     * hardcoded FFT for order = 1
     * @tparam is_forward
     * @tparam F
     * @tparam InPtr
     * @tparam OutPtr
     * @param in
     * @param out
     */
    template <bool is_forward, typename F, typename InPtr, typename OutPtr>
    inline void callback_order_1(InPtr in, OutPtr out) {
        F r0, i0, r1, i1;
        load_scalar<is_forward>(in, r0, i0);
        load_scalar<is_forward>(in.shift(InPtr::get_complex_offset(1)), r1, i1);

        store_scalar<is_forward>(out, r0 + r1, i0 + i1);
        store_scalar<is_forward>(out.shift(OutPtr::get_complex_offset(1)), r0 - r1, i0 - i1);
    }

    /**
     * hardcoded FFT for order = 2
     * @tparam is_forward
     * @tparam F
     * @tparam InPtr
     * @tparam OutPtr
     * @param in
     * @param out
     */
    template <bool is_forward, typename F, typename InPtr, typename OutPtr>
    inline void callback_order_2(InPtr in, OutPtr out) {
        F r0, i0, r1, i1, r2, i2, r3, i3;

        load_scalar<is_forward>(in, r0, i0);
        load_scalar<is_forward>(in.shift(InPtr::get_complex_offset(1)), r1, i1);
        load_scalar<is_forward>(in.shift(InPtr::get_complex_offset(2)), r2, i2);
        load_scalar<is_forward>(in.shift(InPtr::get_complex_offset(3)), r3, i3);

        const auto t0_r = r0 + r2, t0_i = i0 + i2;
        const auto t1_r = r0 - r2, t1_i = i0 - i2;
        const auto t2_r = r1 + r3, t2_i = i1 + i3;
        const auto t3_r = r1 - r3, t3_i = i1 - i3;

        store_scalar<is_forward>(out, t0_r + t2_r, t0_i + t2_i);
        store_scalar<is_forward>(out.shift(OutPtr::get_complex_offset(1)), t1_r + t3_i, t1_i - t3_r);
        store_scalar<is_forward>(out.shift(OutPtr::get_complex_offset(2)), t0_r - t2_r, t0_i - t2_i);
        store_scalar<is_forward>(out.shift(OutPtr::get_complex_offset(3)), t1_r - t3_i, t1_i + t3_r);
    }

    /**
     * hardcoded FFT for order = 3
     * @tparam is_forward
     * @tparam F
     * @tparam InPtr
     * @tparam OutPtr
     * @param in
     * @param out
     */
    template <bool is_forward, typename F, typename InPtr, typename OutPtr>
    inline void callback_order_3(InPtr in, OutPtr out) {
        static constexpr F kInvSqrt2 = static_cast<F>(1.0 / std::numbers::sqrt2);

        F x0_r, x0_i, x1_r, x1_i, x2_r, x2_i, x3_r, x3_i;
        F x4_r, x4_i, x5_r, x5_i, x6_r, x6_i, x7_r, x7_i;

        load_scalar<is_forward>(in, x0_r, x0_i);
        load_scalar<is_forward>(in.shift(InPtr::get_complex_offset(1)), x1_r, x1_i);
        load_scalar<is_forward>(in.shift(InPtr::get_complex_offset(2)), x2_r, x2_i);
        load_scalar<is_forward>(in.shift(InPtr::get_complex_offset(3)), x3_r, x3_i);
        load_scalar<is_forward>(in.shift(InPtr::get_complex_offset(4)), x4_r, x4_i);
        load_scalar<is_forward>(in.shift(InPtr::get_complex_offset(5)), x5_r, x5_i);
        load_scalar<is_forward>(in.shift(InPtr::get_complex_offset(6)), x6_r, x6_i);
        load_scalar<is_forward>(in.shift(InPtr::get_complex_offset(7)), x7_r, x7_i);

        const auto t0_r = x0_r + x4_r, t0_i = x0_i + x4_i;
        const auto t1_r = x0_r - x4_r, t1_i = x0_i - x4_i;
        const auto t2_r = x2_r + x6_r, t2_i = x2_i + x6_i;
        const auto t3_r = x2_r - x6_r, t3_i = x2_i - x6_i;

        const auto u0_r = x1_r + x5_r, u0_i = x1_i + x5_i;
        const auto u1_r = x1_r - x5_r, u1_i = x1_i - x5_i;
        const auto u2_r = x3_r + x7_r, u2_i = x3_i + x7_i;
        const auto u3_r = x3_r - x7_r, u3_i = x3_i - x7_i;

        const auto y00_r = t0_r + t2_r, y00_i = t0_i + t2_i;
        const auto y02_r = t0_r - t2_r, y02_i = t0_i - t2_i;
        const auto y10_r = u0_r + u2_r, y10_i = u0_i + u2_i;
        const auto y12_r = u0_r - u2_r, y12_i = u0_i - u2_i;

        const auto y01_r = t1_r + t3_i, y01_i = t1_i - t3_r;
        const auto y03_r = t1_r - t3_i, y03_i = t1_i + t3_r;
        const auto y11_r = u1_r + u3_i, y11_i = u1_i - u3_r;
        const auto y13_r = u1_r - u3_i, y13_i = u1_i + u3_r;

        const auto v0_r = y10_r, v0_i = y10_i;
        const auto v1_r = (y11_r + y11_i) * kInvSqrt2;
        const auto v1_i = (y11_i - y11_r) * kInvSqrt2;
        const auto v2_r = y12_i, v2_i = -y12_r;
        const auto v3_r = (y13_i - y13_r) * kInvSqrt2;
        const auto v3_i = -(y13_r + y13_i) * kInvSqrt2;

        store_scalar<is_forward>(out, y00_r + v0_r, y00_i + v0_i);
        store_scalar<is_forward>(out.shift(OutPtr::get_complex_offset(1)), y01_r + v1_r, y01_i + v1_i);
        store_scalar<is_forward>(out.shift(OutPtr::get_complex_offset(2)), y02_r + v2_r, y02_i + v2_i);
        store_scalar<is_forward>(out.shift(OutPtr::get_complex_offset(3)), y03_r + v3_r, y03_i + v3_i);
        store_scalar<is_forward>(out.shift(OutPtr::get_complex_offset(4)), y00_r - v0_r, y00_i - v0_i);
        store_scalar<is_forward>(out.shift(OutPtr::get_complex_offset(5)), y01_r - v1_r, y01_i - v1_i);
        store_scalar<is_forward>(out.shift(OutPtr::get_complex_offset(6)), y02_r - v2_r, y02_i - v2_i);
        store_scalar<is_forward>(out.shift(OutPtr::get_complex_offset(7)), y03_r - v3_r, y03_i - v3_i);
    }

    /**
     * hardcoded FFT for order = 4
     * @tparam is_forward
     * @tparam F
     * @tparam InPtr
     * @tparam OutPtr
     * @param in
     * @param out
     * @param w_r_base
     * @param w_i_base
     */
    template <bool is_forward, typename F, typename InPtr, typename OutPtr>
    inline void callback_order_4(InPtr in, OutPtr out, const F* w_r_base, const F* w_i_base) {
        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        HWY_ALIGN F tmp_r[16];
        HWY_ALIGN F tmp_i[16];

        if constexpr (lanes == 8) {
            hn::FixedTag<F, 4> d4;
            {
                hn::Vec<decltype(d)> x0_r, x0_i, x1_r, x1_i;
                load_complex<is_forward>(d, in, x0_r, x0_i);
                load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(8)), x1_r, x1_i);

                const auto t02_r = hn::Add(x0_r, x1_r), t02_i = hn::Add(x0_i, x1_i);
                const auto t13_r = hn::Sub(x0_r, x1_r), t13_i = hn::Sub(x0_i, x1_i);

                const auto A_r = hn::ConcatLowerLower(d, t13_r, t02_r);
                const auto A_i = hn::ConcatLowerLower(d, t13_i, t02_i);
                const auto B_r = hn::ConcatUpperUpper(d, t13_i, t02_r);
                const auto B_i = hn::ConcatUpperUpper(d, t13_r, t02_i);

                const auto out01_r = hn::Add(A_r, B_r);
                const auto out23_r = hn::Sub(A_r, B_r);
                const auto sum_i = hn::Add(A_i, B_i);
                const auto diff_i = hn::Sub(A_i, B_i);

                hn::StoreInterleaved4(hn::LowerHalf(d4, out01_r), hn::UpperHalf(d4, out01_r),
                                      hn::LowerHalf(d4, out23_r),
                                      hn::UpperHalf(d4, out23_r), d4, tmp_r);
                hn::StoreInterleaved4(hn::LowerHalf(d4, sum_i), hn::UpperHalf(d4, diff_i), hn::LowerHalf(d4, diff_i),
                                      hn::UpperHalf(d4, sum_i), d4, tmp_i);
            }
            hn::Vec<decltype(d4)> t1_r, t1_i;
            {
                const auto r1 = hn::Load(d4, tmp_r + 4), i1 = hn::Load(d4, tmp_i + 4);
                const auto w1_r = hn::Load(d4, w_r_base), w1_i = hn::Load(d4, w_i_base);
                t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
                t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));
            }
            hn::Vec<decltype(d4)> t3_r, t3_i;
            {
                const auto r3 = hn::Load(d4, tmp_r + 12), i3 = hn::Load(d4, tmp_i + 12);
                const auto w3_r = hn::Load(d4, w_r_base + 8), w3_i = hn::Load(d4, w_i_base + 8);
                t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
                t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));
            }
            hn::Vec<decltype(d4)> t2_r, t2_i;
            {
                const auto r2 = hn::Load(d4, tmp_r + 8), i2 = hn::Load(d4, tmp_i + 8);
                const auto w2_r = hn::Load(d4, w_r_base + 4), w2_i = hn::Load(d4, w_i_base + 4);
                t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
                t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));
            }

            const auto t0_r = hn::Load(d4, tmp_r), t0_i = hn::Load(d4, tmp_i);

            const auto s0_r = hn::Add(t0_r, t2_r), s0_i = hn::Add(t0_i, t2_i);
            const auto s1_r = hn::Sub(t0_r, t2_r), s1_i = hn::Sub(t0_i, t2_i);
            const auto s2_r = hn::Add(t1_r, t3_r), s2_i = hn::Add(t1_i, t3_i);
            const auto s3_r = hn::Sub(t1_r, t3_r), s3_i = hn::Sub(t1_i, t3_i);
            {
                const auto f0_r = hn::Add(s0_r, s2_r), f0_i = hn::Add(s0_i, s2_i);
                const auto f1_r = hn::Add(s1_r, s3_i), f1_i = hn::Sub(s1_i, s3_r);
                store_complex<is_forward>(d, out,
                                          hn::Combine(d, f1_r, f0_r), hn::Combine(d, f1_i, f0_i));
            }
            {
                const auto f2_r = hn::Sub(s0_r, s2_r), f2_i = hn::Sub(s0_i, s2_i);
                const auto f3_r = hn::Sub(s1_r, s3_i), f3_i = hn::Add(s1_i, s3_r);
                store_complex<is_forward>(d, out.shift(OutPtr::get_complex_offset(8)),
                                          hn::Combine(d, f3_r, f2_r), hn::Combine(d, f3_i, f2_i));
            }
        } else if constexpr (lanes <= 4) {
            for (size_t i = 0; i < 4; i += lanes) {
                hn::Vec<decltype(d)> x0_r, x0_i, x1_r, x1_i, x2_r, x2_i, x3_r, x3_i;

                load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(i)), x0_r, x0_i);
                load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(i + 8)), x2_r, x2_i);

                const auto t0_r = hn::Add(x0_r, x2_r), t0_i = hn::Add(x0_i, x2_i);
                const auto t1_r = hn::Sub(x0_r, x2_r), t1_i = hn::Sub(x0_i, x2_i);

                load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(i + 4)), x1_r, x1_i);
                load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(i + 12)), x3_r, x3_i);

                const auto t2_r = hn::Add(x1_r, x3_r), t2_i = hn::Add(x1_i, x3_i);
                const auto t3_r = hn::Sub(x1_r, x3_r), t3_i = hn::Sub(x1_i, x3_i);

                const auto out0_r = hn::Add(t0_r, t2_r), out0_i = hn::Add(t0_i, t2_i);
                const auto out2_r = hn::Sub(t0_r, t2_r), out2_i = hn::Sub(t0_i, t2_i);
                const auto out1_r = hn::Add(t1_r, t3_i), out1_i = hn::Sub(t1_i, t3_r);
                const auto out3_r = hn::Sub(t1_r, t3_i), out3_i = hn::Add(t1_i, t3_r);

                hn::StoreInterleaved4(out0_r, out1_r, out2_r, out3_r, d, tmp_r + i * 4);
                hn::StoreInterleaved4(out0_i, out1_i, out2_i, out3_i, d, tmp_i + i * 4);
            }
            for (size_t i = 0; i < 4; i += lanes) {
                const auto i1 = hn::Load(d, tmp_i + 4 + i), r1 = hn::Load(d, tmp_r + 4 + i);
                const auto w1_r = hn::Load(d, w_r_base + i), w1_i = hn::Load(d, w_i_base + i);
                const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
                const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

                const auto i3 = hn::Load(d, tmp_i + 12 + i), r3 = hn::Load(d, tmp_r + 12 + i);
                const auto w3_r = hn::Load(d, w_r_base + 8 + i), w3_i = hn::Load(d, w_i_base + 8 + i);
                const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
                const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

                const auto s2_r = hn::Add(t1_r, t3_r), s2_i = hn::Add(t1_i, t3_i);
                const auto s3_r = hn::Sub(t1_r, t3_r), s3_i = hn::Sub(t1_i, t3_i);

                const auto i2 = hn::Load(d, tmp_i + 8 + i), r2 = hn::Load(d, tmp_r + 8 + i);
                const auto w2_r = hn::Load(d, w_r_base + 4 + i), w2_i = hn::Load(d, w_i_base + 4 + i);
                const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
                const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

                const auto r0 = hn::Load(d, tmp_r + i), i0 = hn::Load(d, tmp_i + i);
                const auto s0_r = hn::Add(r0, t2_r), s0_i = hn::Add(i0, t2_i);
                const auto s1_r = hn::Sub(r0, t2_r), s1_i = hn::Sub(i0, t2_i);

                store_complex<is_forward>(d, out.shift(OutPtr::get_complex_offset(i)),
                                          hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i));
                store_complex<is_forward>(d, out.shift(OutPtr::get_complex_offset(i + 8)),
                                          hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i));
                store_complex<is_forward>(d, out.shift(OutPtr::get_complex_offset(i + 4)),
                                          hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r));
                store_complex<is_forward>(d, out.shift(OutPtr::get_complex_offset(i + 12)),
                                          hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r));
            }
        }
    }

    /**
     * hardcoded FFT for order = 5
     * @tparam is_forward
     * @tparam F
     * @tparam InPtr
     * @tparam OutPtr
     * @param in
     * @param out
     * @param w_r_base
     * @param w_i_base
     */
    template <bool is_forward, typename F, typename InPtr, typename OutPtr>
    inline void callback_order_5(InPtr in, OutPtr out, const F* w_r_base, const F* w_i_base) {
        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);
        static constexpr F kInvSqrt2 = static_cast<F>(1.0 / std::numbers::sqrt2);

        HWY_ALIGN F tmp_r[32];
        HWY_ALIGN F tmp_i[32];

        if constexpr (lanes == 8) {
            const hn::FixedTag<F, 4> d4;
            const auto inv_sqrt2 = hn::Set(d4, kInvSqrt2);

            hn::Vec<decltype(d)> a_r, a_i, b_r, b_i;
            load_complex<is_forward>(d, in, a_r, a_i);
            load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(16)), b_r, b_i);
            const auto vec_t0_r = hn::Add(a_r, b_r), vec_t0_i = hn::Add(a_i, b_i);
            const auto vec_t1_r = hn::Sub(a_r, b_r), vec_t1_i = hn::Sub(a_i, b_i);

            load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(8)), a_r, a_i);
            load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(24)), b_r, b_i);
            const auto vec_t2_r = hn::Add(a_r, b_r), vec_t2_i = hn::Add(a_i, b_i);
            const auto vec_t3_r = hn::Sub(a_r, b_r), vec_t3_i = hn::Sub(a_i, b_i);

            const auto vec_y00_r = hn::Add(vec_t0_r, vec_t2_r), vec_y00_i = hn::Add(vec_t0_i, vec_t2_i);
            const auto vec_y01_r = hn::Add(vec_t1_r, vec_t3_i), vec_y01_i = hn::Sub(vec_t1_i, vec_t3_r);
            const auto vec_y02_r = hn::Sub(vec_t0_r, vec_t2_r), vec_y02_i = hn::Sub(vec_t0_i, vec_t2_i);
            const auto vec_y03_r = hn::Sub(vec_t1_r, vec_t3_i), vec_y03_i = hn::Add(vec_t1_i, vec_t3_r);

            const auto y00_r = hn::LowerHalf(d4, vec_y00_r), y00_i = hn::LowerHalf(d4, vec_y00_i);
            const auto y01_r = hn::LowerHalf(d4, vec_y01_r), y01_i = hn::LowerHalf(d4, vec_y01_i);
            const auto y02_r = hn::LowerHalf(d4, vec_y02_r), y02_i = hn::LowerHalf(d4, vec_y02_i);
            const auto y03_r = hn::LowerHalf(d4, vec_y03_r), y03_i = hn::LowerHalf(d4, vec_y03_i);

            const auto y10_r = hn::UpperHalf(d4, vec_y00_r), y10_i = hn::UpperHalf(d4, vec_y00_i);
            const auto y11_r = hn::UpperHalf(d4, vec_y01_r), y11_i = hn::UpperHalf(d4, vec_y01_i);
            const auto y12_r = hn::UpperHalf(d4, vec_y02_r), y12_i = hn::UpperHalf(d4, vec_y02_i);
            const auto y13_r = hn::UpperHalf(d4, vec_y03_r), y13_i = hn::UpperHalf(d4, vec_y03_i);

            const auto v0_r = y10_r, v0_i = y10_i;
            const auto v1_r = hn::Mul(hn::Add(y11_r, y11_i), inv_sqrt2),
                v1_i = hn::Mul(hn::Sub(y11_i, y11_r), inv_sqrt2);
            const auto v2_r = y12_i, v2_i = hn::Neg(y12_r);
            const auto v3_r = hn::Mul(hn::Sub(y13_i, y13_r), inv_sqrt2),
                v3_i = hn::Mul(hn::Neg(hn::Add(y13_r, y13_i)), inv_sqrt2);

            const auto z00_r = hn::Add(y00_r, v0_r), z00_i = hn::Add(y00_i, v0_i);
            const auto z01_r = hn::Add(y01_r, v1_r), z01_i = hn::Add(y01_i, v1_i);
            const auto z02_r = hn::Add(y02_r, v2_r), z02_i = hn::Add(y02_i, v2_i);
            const auto z03_r = hn::Add(y03_r, v3_r), z03_i = hn::Add(y03_i, v3_i);

            const auto z10_r = hn::Sub(y00_r, v0_r), z10_i = hn::Sub(y00_i, v0_i);
            const auto z11_r = hn::Sub(y01_r, v1_r), z11_i = hn::Sub(y01_i, v1_i);
            const auto z12_r = hn::Sub(y02_r, v2_r), z12_i = hn::Sub(y02_i, v2_i);
            const auto z13_r = hn::Sub(y03_r, v3_r), z13_i = hn::Sub(y03_i, v3_i);

            const auto lower_r0 = hn::InterleaveLower(d4, z00_r, z10_r);
            const auto upper_r0 = hn::InterleaveUpper(d4, z00_r, z10_r);
            const auto lower_r1 = hn::InterleaveLower(d4, z01_r, z11_r);
            const auto upper_r1 = hn::InterleaveUpper(d4, z01_r, z11_r);
            const auto lower_r2 = hn::InterleaveLower(d4, z02_r, z12_r);
            const auto upper_r2 = hn::InterleaveUpper(d4, z02_r, z12_r);
            const auto lower_r3 = hn::InterleaveLower(d4, z03_r, z13_r);
            const auto upper_r3 = hn::InterleaveUpper(d4, z03_r, z13_r);

            const auto lower_i0 = hn::InterleaveLower(d4, z00_i, z10_i);
            const auto upper_i0 = hn::InterleaveUpper(d4, z00_i, z10_i);
            const auto lower_i1 = hn::InterleaveLower(d4, z01_i, z11_i);
            const auto upper_i1 = hn::InterleaveUpper(d4, z01_i, z11_i);
            const auto lower_i2 = hn::InterleaveLower(d4, z02_i, z12_i);
            const auto upper_i2 = hn::InterleaveUpper(d4, z02_i, z12_i);
            const auto lower_i3 = hn::InterleaveLower(d4, z03_i, z13_i);
            const auto upper_i3 = hn::InterleaveUpper(d4, z03_i, z13_i);

            hn::StoreInterleaved4(lower_r0, lower_r1, lower_r2, lower_r3, d4, tmp_r);
            hn::StoreInterleaved4(upper_r0, upper_r1, upper_r2, upper_r3, d4, tmp_r + 16);
            hn::StoreInterleaved4(lower_i0, lower_i1, lower_i2, lower_i3, d4, tmp_i);
            hn::StoreInterleaved4(upper_i0, upper_i1, upper_i2, upper_i3, d4, tmp_i + 16);

            const auto i1 = hn::Load(d, tmp_i + 8), r1 = hn::Load(d, tmp_r + 8);
            const auto w1_r = hn::Load(d, w_r_base), w1_i = hn::Load(d, w_i_base);
            const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
            const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

            const auto i3 = hn::Load(d, tmp_i + 24), r3 = hn::Load(d, tmp_r + 24);
            const auto w3_r = hn::Load(d, w_r_base + 16), w3_i = hn::Load(d, w_i_base + 16);
            const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
            const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

            const auto s2_r = hn::Add(t1_r, t3_r), s2_i = hn::Add(t1_i, t3_i);
            const auto s3_r = hn::Sub(t1_r, t3_r), s3_i = hn::Sub(t1_i, t3_i);

            const auto i2 = hn::Load(d, tmp_i + 16), r2 = hn::Load(d, tmp_r + 16);
            const auto w2_r = hn::Load(d, w_r_base + 8), w2_i = hn::Load(d, w_i_base + 8);
            const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
            const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

            const auto r0 = hn::Load(d, tmp_r), i0 = hn::Load(d, tmp_i);
            const auto s0_r = hn::Add(r0, t2_r), s0_i = hn::Add(i0, t2_i);
            const auto s1_r = hn::Sub(r0, t2_r), s1_i = hn::Sub(i0, t2_i);

            store_complex<is_forward>(d, out,
                                      hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i));
            store_complex<is_forward>(d, out.shift(OutPtr::get_complex_offset(8)),
                                      hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r));
            store_complex<is_forward>(d, out.shift(OutPtr::get_complex_offset(16)),
                                      hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i));
            store_complex<is_forward>(d, out.shift(OutPtr::get_complex_offset(24)),
                                      hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r));

        } else if constexpr (lanes <= 4) {
            const auto inv_sqrt2 = hn::Set(d, kInvSqrt2);
            if constexpr (lanes == 4) {
                hn::Vec<decltype(d)> a_r, a_i, b_r, b_i;

                load_complex<is_forward>(d, in, a_r, a_i);
                load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(16)), b_r, b_i);
                const auto t0_r = hn::Add(a_r, b_r), t0_i = hn::Add(a_i, b_i);
                const auto t1_r = hn::Sub(a_r, b_r), t1_i = hn::Sub(a_i, b_i);

                load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(8)), a_r, a_i);
                load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(24)), b_r, b_i);
                const auto t2_r = hn::Add(a_r, b_r), t2_i = hn::Add(a_i, b_i);
                const auto t3_r = hn::Sub(a_r, b_r), t3_i = hn::Sub(a_i, b_i);

                const auto y00_r = hn::Add(t0_r, t2_r), y00_i = hn::Add(t0_i, t2_i);
                const auto y01_r = hn::Add(t1_r, t3_i), y01_i = hn::Sub(t1_i, t3_r);
                const auto y02_r = hn::Sub(t0_r, t2_r), y02_i = hn::Sub(t0_i, t2_i);
                const auto y03_r = hn::Sub(t1_r, t3_i), y03_i = hn::Add(t1_i, t3_r);

                load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(4)), a_r, a_i);
                load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(20)), b_r, b_i);
                const auto u0_r = hn::Add(a_r, b_r), u0_i = hn::Add(a_i, b_i);
                const auto u1_r = hn::Sub(a_r, b_r), u1_i = hn::Sub(a_i, b_i);

                load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(12)), a_r, a_i);
                load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(28)), b_r, b_i);
                const auto u2_r = hn::Add(a_r, b_r), u2_i = hn::Add(a_i, b_i);
                const auto u3_r = hn::Sub(a_r, b_r), u3_i = hn::Sub(a_i, b_i);

                const auto y10_r = hn::Add(u0_r, u2_r), y10_i = hn::Add(u0_i, u2_i);
                const auto y11_r = hn::Add(u1_r, u3_i), y11_i = hn::Sub(u1_i, u3_r);
                const auto y12_r = hn::Sub(u0_r, u2_r), y12_i = hn::Sub(u0_i, u2_i);
                const auto y13_r = hn::Sub(u1_r, u3_i), y13_i = hn::Add(u1_i, u3_r);

                const auto v0_r = y10_r;
                const auto v0_i = y10_i;
                const auto v1_r = hn::Mul(hn::Add(y11_r, y11_i), inv_sqrt2);
                const auto v1_i = hn::Mul(hn::Sub(y11_i, y11_r), inv_sqrt2);
                const auto v2_r = y12_i;
                const auto v2_i = hn::Neg(y12_r);
                const auto v3_r = hn::Mul(hn::Sub(y13_i, y13_r), inv_sqrt2);
                const auto v3_i = hn::Mul(hn::Neg(hn::Add(y13_r, y13_i)), inv_sqrt2);

                const auto z00_r = hn::Add(y00_r, v0_r), z00_i = hn::Add(y00_i, v0_i);
                const auto z01_r = hn::Add(y01_r, v1_r), z01_i = hn::Add(y01_i, v1_i);
                const auto z02_r = hn::Add(y02_r, v2_r), z02_i = hn::Add(y02_i, v2_i);
                const auto z03_r = hn::Add(y03_r, v3_r), z03_i = hn::Add(y03_i, v3_i);

                const auto z10_r = hn::Sub(y00_r, v0_r), z10_i = hn::Sub(y00_i, v0_i);
                const auto z11_r = hn::Sub(y01_r, v1_r), z11_i = hn::Sub(y01_i, v1_i);
                const auto z12_r = hn::Sub(y02_r, v2_r), z12_i = hn::Sub(y02_i, v2_i);
                const auto z13_r = hn::Sub(y03_r, v3_r), z13_i = hn::Sub(y03_i, v3_i);

                const auto lower_r0 = hn::InterleaveLower(d, z00_r, z10_r);
                const auto upper_r0 = hn::InterleaveUpper(d, z00_r, z10_r);
                const auto lower_r1 = hn::InterleaveLower(d, z01_r, z11_r);
                const auto upper_r1 = hn::InterleaveUpper(d, z01_r, z11_r);
                const auto lower_r2 = hn::InterleaveLower(d, z02_r, z12_r);
                const auto upper_r2 = hn::InterleaveUpper(d, z02_r, z12_r);
                const auto lower_r3 = hn::InterleaveLower(d, z03_r, z13_r);
                const auto upper_r3 = hn::InterleaveUpper(d, z03_r, z13_r);

                const auto lower_i0 = hn::InterleaveLower(d, z00_i, z10_i);
                const auto upper_i0 = hn::InterleaveUpper(d, z00_i, z10_i);
                const auto lower_i1 = hn::InterleaveLower(d, z01_i, z11_i);
                const auto upper_i1 = hn::InterleaveUpper(d, z01_i, z11_i);
                const auto lower_i2 = hn::InterleaveLower(d, z02_i, z12_i);
                const auto upper_i2 = hn::InterleaveUpper(d, z02_i, z12_i);
                const auto lower_i3 = hn::InterleaveLower(d, z03_i, z13_i);
                const auto upper_i3 = hn::InterleaveUpper(d, z03_i, z13_i);

                hn::StoreInterleaved4(lower_r0, lower_r1, lower_r2, lower_r3, d, tmp_r);
                hn::StoreInterleaved4(upper_r0, upper_r1, upper_r2, upper_r3, d, tmp_r + lanes * 4);
                hn::StoreInterleaved4(lower_i0, lower_i1, lower_i2, lower_i3, d, tmp_i);
                hn::StoreInterleaved4(upper_i0, upper_i1, upper_i2, upper_i3, d, tmp_i + lanes * 4);
            } else {
                for (size_t idx = 0; idx < 4; idx += lanes) {
                    hn::Vec<decltype(d)> a_r, a_i, b_r, b_i;

                    load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(idx)), a_r, a_i);
                    load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(idx + 16)), b_r, b_i);
                    const auto t0_r = hn::Add(a_r, b_r), t0_i = hn::Add(a_i, b_i);
                    const auto t1_r = hn::Sub(a_r, b_r), t1_i = hn::Sub(a_i, b_i);

                    load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(idx + 8)), a_r, a_i);
                    load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(idx + 24)), b_r, b_i);
                    const auto t2_r = hn::Add(a_r, b_r), t2_i = hn::Add(a_i, b_i);
                    const auto t3_r = hn::Sub(a_r, b_r), t3_i = hn::Sub(a_i, b_i);

                    const auto y00_r = hn::Add(t0_r, t2_r), y00_i = hn::Add(t0_i, t2_i);
                    const auto y01_r = hn::Add(t1_r, t3_i), y01_i = hn::Sub(t1_i, t3_r);
                    const auto y02_r = hn::Sub(t0_r, t2_r), y02_i = hn::Sub(t0_i, t2_i);
                    const auto y03_r = hn::Sub(t1_r, t3_i), y03_i = hn::Add(t1_i, t3_r);

                    load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(idx + 4)), a_r, a_i);
                    load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(idx + 20)), b_r, b_i);
                    const auto u0_r = hn::Add(a_r, b_r), u0_i = hn::Add(a_i, b_i);
                    const auto u1_r = hn::Sub(a_r, b_r), u1_i = hn::Sub(a_i, b_i);

                    load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(idx + 12)), a_r, a_i);
                    load_complex<is_forward>(d, in.shift(InPtr::get_complex_offset(idx + 28)), b_r, b_i);
                    const auto u2_r = hn::Add(a_r, b_r), u2_i = hn::Add(a_i, b_i);
                    const auto u3_r = hn::Sub(a_r, b_r), u3_i = hn::Sub(a_i, b_i);

                    const auto y10_r = hn::Add(u0_r, u2_r), y10_i = hn::Add(u0_i, u2_i);
                    const auto y11_r = hn::Add(u1_r, u3_i), y11_i = hn::Sub(u1_i, u3_r);
                    const auto y12_r = hn::Sub(u0_r, u2_r), y12_i = hn::Sub(u0_i, u2_i);
                    const auto y13_r = hn::Sub(u1_r, u3_i), y13_i = hn::Add(u1_i, u3_r);

                    const auto v0_r = y10_r;
                    const auto v0_i = y10_i;
                    const auto v1_r = hn::Mul(hn::Add(y11_r, y11_i), inv_sqrt2);
                    const auto v1_i = hn::Mul(hn::Sub(y11_i, y11_r), inv_sqrt2);
                    const auto v2_r = y12_i;
                    const auto v2_i = hn::Neg(y12_r);
                    const auto v3_r = hn::Mul(hn::Sub(y13_i, y13_r), inv_sqrt2);
                    const auto v3_i = hn::Mul(hn::Neg(hn::Add(y13_r, y13_i)), inv_sqrt2);

                    const auto z00_r = hn::Add(y00_r, v0_r), z00_i = hn::Add(y00_i, v0_i);
                    const auto z01_r = hn::Add(y01_r, v1_r), z01_i = hn::Add(y01_i, v1_i);
                    const auto z02_r = hn::Add(y02_r, v2_r), z02_i = hn::Add(y02_i, v2_i);
                    const auto z03_r = hn::Add(y03_r, v3_r), z03_i = hn::Add(y03_i, v3_i);

                    const auto z10_r = hn::Sub(y00_r, v0_r), z10_i = hn::Sub(y00_i, v0_i);
                    const auto z11_r = hn::Sub(y01_r, v1_r), z11_i = hn::Sub(y01_i, v1_i);
                    const auto z12_r = hn::Sub(y02_r, v2_r), z12_i = hn::Sub(y02_i, v2_i);
                    const auto z13_r = hn::Sub(y03_r, v3_r), z13_i = hn::Sub(y03_i, v3_i);

                    const auto lower_r0 = hn::InterleaveLower(d, z00_r, z10_r);
                    const auto upper_r0 = hn::InterleaveUpper(d, z00_r, z10_r);
                    const auto lower_r1 = hn::InterleaveLower(d, z01_r, z11_r);
                    const auto upper_r1 = hn::InterleaveUpper(d, z01_r, z11_r);
                    const auto lower_r2 = hn::InterleaveLower(d, z02_r, z12_r);
                    const auto upper_r2 = hn::InterleaveUpper(d, z02_r, z12_r);
                    const auto lower_r3 = hn::InterleaveLower(d, z03_r, z13_r);
                    const auto upper_r3 = hn::InterleaveUpper(d, z03_r, z13_r);

                    const auto lower_i0 = hn::InterleaveLower(d, z00_i, z10_i);
                    const auto upper_i0 = hn::InterleaveUpper(d, z00_i, z10_i);
                    const auto lower_i1 = hn::InterleaveLower(d, z01_i, z11_i);
                    const auto upper_i1 = hn::InterleaveUpper(d, z01_i, z11_i);
                    const auto lower_i2 = hn::InterleaveLower(d, z02_i, z12_i);
                    const auto upper_i2 = hn::InterleaveUpper(d, z02_i, z12_i);
                    const auto lower_i3 = hn::InterleaveLower(d, z03_i, z13_i);
                    const auto upper_i3 = hn::InterleaveUpper(d, z03_i, z13_i);

                    hn::StoreInterleaved4(lower_r0, lower_r1, lower_r2, lower_r3, d, tmp_r + idx * 8);
                    hn::StoreInterleaved4(upper_r0, upper_r1, upper_r2, upper_r3, d, tmp_r + idx * 8 + lanes * 4);
                    hn::StoreInterleaved4(lower_i0, lower_i1, lower_i2, lower_i3, d, tmp_i + idx * 8);
                    hn::StoreInterleaved4(upper_i0, upper_i1, upper_i2, upper_i3, d, tmp_i + idx * 8 + lanes * 4);
                }
            }

            static constexpr size_t off1 = (sizeof(F) == 8 && lanes == 4) ? 16 : 8;
            static constexpr size_t off2 = (sizeof(F) == 8 && lanes == 4) ? 8 : 16;

            for (size_t k = 0; k < 8; k += lanes) {
                const auto i1 = hn::Load(d, tmp_i + off1 + k), r1 = hn::Load(d, tmp_r + off1 + k);
                const auto w1_r = hn::Load(d, w_r_base + k), w1_i = hn::Load(d, w_i_base + k);
                const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
                const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

                const auto i3 = hn::Load(d, tmp_i + 24 + k), r3 = hn::Load(d, tmp_r + 24 + k);
                const auto w3_r = hn::Load(d, w_r_base + 16 + k), w3_i = hn::Load(d, w_i_base + 16 + k);
                const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
                const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

                const auto s2_r = hn::Add(t1_r, t3_r), s2_i = hn::Add(t1_i, t3_i);
                const auto s3_r = hn::Sub(t1_r, t3_r), s3_i = hn::Sub(t1_i, t3_i);

                const auto i2 = hn::Load(d, tmp_i + off2 + k), r2 = hn::Load(d, tmp_r + off2 + k);
                const auto w2_r = hn::Load(d, w_r_base + 8 + k), w2_i = hn::Load(d, w_i_base + 8 + k);
                const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
                const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

                const auto r0 = hn::Load(d, tmp_r + k), i0 = hn::Load(d, tmp_i + k);
                const auto s0_r = hn::Add(r0, t2_r), s0_i = hn::Add(i0, t2_i);
                const auto s1_r = hn::Sub(r0, t2_r), s1_i = hn::Sub(i0, t2_i);

                store_complex<is_forward>(d, out.shift(OutPtr::get_complex_offset(k)),
                                          hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i));
                store_complex<is_forward>(d, out.shift(OutPtr::get_complex_offset(k + 8)),
                                          hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r));
                store_complex<is_forward>(d, out.shift(OutPtr::get_complex_offset(k + 16)),
                                          hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i));
                store_complex<is_forward>(d, out.shift(OutPtr::get_complex_offset(k + 24)),
                                          hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r));
            }
        }
    }
}
