#pragma once

#include "zldsp_fft_common_init.hpp"
#include "zldsp_fft_common_radix.hpp"
#include "zldsp_fft_common_kernel.hpp"

namespace zldsp::fft::common {
    namespace hn = hwy::HWY_NAMESPACE;

    /**
     * execute Stockham CFFT
     * @tparam is_forward
     * @tparam F
     * @tparam InPtr
     * @tparam OutPtr
     * @param cfft_order
     * @param stride
     * @param workspace
     * @param twiddles
     * @param twiddles_shift
     * @param stages
     * @param in_ptr
     * @param out_ptr
     */
    template <bool is_forward, typename F, typename InPtr, typename OutPtr>
    HWY_INLINE void execute_stockham_cfft(
        const size_t cfft_order,
        const size_t stride,
        F* HWY_RESTRICT workspace,
        const F* HWY_RESTRICT twiddles,
        const std::vector<size_t>& twiddles_shift,
        const std::vector<StageType>& stages,
        InPtr in_ptr, OutPtr out_ptr) {
        const size_t cfft_size = 1 << cfft_order;
        switch (cfft_order) {
        case 0: {
            common::callback_order_0<is_forward, F>(in_ptr, out_ptr);
            return;
        }
        case 1: {
            common::callback_order_1<is_forward, F>(in_ptr, out_ptr);
            return;
        }
        case 2: {
            common::callback_order_2<is_forward, F>(in_ptr, out_ptr);
            return;
        }
        case 3: {
            common::callback_order_3<is_forward, F>(in_ptr, out_ptr);
            return;
        }
        case 4: {
            common::callback_order_4<is_forward, F>(in_ptr, out_ptr, twiddles, twiddles + 60);
            return;
        }
        case 5: {
            common::callback_order_5<is_forward, F>(in_ptr, out_ptr, twiddles, twiddles + 120);
            return;
        }
        case 6: {
            F* HWY_RESTRICT in_aosoa = workspace;
            F* HWY_RESTRICT out_aosoa = workspace + 2 * stride;
            const F* HWY_RESTRICT w0 = twiddles;
            common::radix4_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, 64);
            common::radix4_width4_aosoa(out_aosoa, in_aosoa, 64, w0);
            const F* HWY_RESTRICT w1 = w0 + twiddles_shift[1];
            common::radix4_last_pass_fused_aosoa<is_forward>(in_aosoa, out_ptr, 64, 16, w1);
            return;
        }
        case 7: {
            F* HWY_RESTRICT in_aosoa = workspace;
            F* HWY_RESTRICT out_aosoa = workspace + 2 * stride;
            const F* HWY_RESTRICT w0 = twiddles;
            common::radix8_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, 128);
            common::radix4_aosoa(out_aosoa, in_aosoa, 128, 8, w0);
            const F* HWY_RESTRICT w1 = w0 + twiddles_shift[1];
            common::radix4_last_pass_fused_aosoa<is_forward>(in_aosoa, out_ptr, 128, 32, w1);
            return;
        }
        case 8: {
            F* HWY_RESTRICT in_aosoa = workspace;
            F* HWY_RESTRICT out_aosoa = workspace + 2 * stride;
            const F* HWY_RESTRICT w0 = twiddles;
            common::radix4_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, 256);
            common::radix4_width4_aosoa(out_aosoa, in_aosoa, 256, w0);
            const F* HWY_RESTRICT w1 = w0 + twiddles_shift[1];
            common::radix4_aosoa(in_aosoa, out_aosoa, 256, 16, w1);
            const F* HWY_RESTRICT w2 = w1 + twiddles_shift[2];
            common::radix4_last_pass_fused_aosoa<is_forward>(out_aosoa, out_ptr, 256, 64, w2);
            return;
        }
        case 9: {
            F* HWY_RESTRICT in_aosoa = workspace;
            F* HWY_RESTRICT out_aosoa = workspace + 2 * stride;
            const F* HWY_RESTRICT w0 = twiddles;
            common::radix8_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, 512);
            common::radix4_aosoa(out_aosoa, in_aosoa, 512, 8, w0);
            const F* HWY_RESTRICT w1 = w0 + twiddles_shift[1];
            common::radix4_aosoa(in_aosoa, out_aosoa, 512, 32, w1);
            const F* HWY_RESTRICT w2 = w1 + twiddles_shift[2];
            common::radix4_last_pass_fused_aosoa<is_forward>(out_aosoa, out_ptr, 512, 128, w2);
            return;
        }
        case 10: {
            F* HWY_RESTRICT in_aosoa = workspace;
            F* HWY_RESTRICT out_aosoa = workspace + 2 * stride;
            const F* HWY_RESTRICT w0 = twiddles;
            common::radix4_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, 1024);
            common::radix4_width4_aosoa(out_aosoa, in_aosoa, 1024, w0);
            const F* HWY_RESTRICT w1 = w0 + twiddles_shift[1];
            common::radix4_aosoa(in_aosoa, out_aosoa, 1024, 16, w1);
            const F* HWY_RESTRICT w2 = w1 + twiddles_shift[2];
            common::radix4_aosoa(out_aosoa, in_aosoa, 1024, 64, w2);
            const F* HWY_RESTRICT w3 = w2 + twiddles_shift[3];
            common::radix4_last_pass_fused_aosoa<is_forward>(in_aosoa, out_ptr, 1024, 256, w3);
            return;
        }
        default:
            break;
        }

        F* HWY_RESTRICT in_aosoa = workspace;
        F* HWY_RESTRICT out_aosoa = workspace + 2 * stride;
        const F* HWY_RESTRICT w_ptr = twiddles;

        if (stages[0] == StageType::kRadix4FirstPass) {
            common::radix4_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, cfft_size);
        } else {
            common::radix8_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, cfft_size);
        }
        std::swap(in_aosoa, out_aosoa);

        size_t width = (stages[0] == StageType::kRadix4FirstPass) ? 4 : 8;
        for (size_t i = 1; i < stages.size() - 1; ++i) {
            switch (stages[i]) {
            case StageType::kRadix4Width4: {
                common::radix4_width4_aosoa(in_aosoa, out_aosoa, cfft_size, w_ptr);
                break;
            }
            case StageType::kRadix4: {
                common::radix4_aosoa(in_aosoa, out_aosoa, cfft_size, width, w_ptr);
                break;
            }
            default:
                break;
            }
            width = width << 2;
            w_ptr += twiddles_shift[i];
            std::swap(in_aosoa, out_aosoa);
        }
        common::radix4_last_pass_fused_aosoa<is_forward>(in_aosoa, out_ptr, cfft_size, width, w_ptr);
    }

    /**
     * execute CFFT
     * @tparam is_forward
     * @tparam F
     * @tparam InPtr
     * @tparam OutPtr
     * @param state
     * @param in_ptr
     * @param out_ptr
     */
    template <bool is_forward, typename F, typename InPtr, typename OutPtr>
    HWY_INLINE void execute_cfft(const CFFTState<F>& state, InPtr in_ptr, OutPtr out_ptr) {
        // small FFT, dispatch to pure Stockham CFFT
        if (state.num_macro_stages == 0) {
            execute_stockham_cfft<is_forward, F, InPtr, OutPtr>(
                state.micro_cfft_order,
                state.micro_stride,
                state.workspace.get(),
                state.micro_twiddles.get(),
                state.micro_twiddles_shift,
                state.micro_stages,
                in_ptr, out_ptr);
            return;
        }
        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);
        static constexpr bool is_soa = std::is_same_v<OutPtr, SoAPtr<F>>;
        // execute macro Cooley-Tukey DIF CFFT
        {
            F* HWY_RESTRICT macro_space = state.workspace.get();
            const F* HWY_RESTRICT w_ptr = state.macro_twiddles.get();
            common::radix4_first_pass_dif_fused_aosoa<is_forward>(in_ptr, macro_space, state.cfft_size, w_ptr);
            w_ptr += state.macro_twiddles_shift[0];

            for (size_t i = 1; i < state.num_macro_stages; ++i) {
                size_t width = state.cfft_size >> (2 * i + 2);
                common::radix4_dif_aosoa_inplace(macro_space, state.cfft_size, width, w_ptr);
                w_ptr += state.macro_twiddles_shift[i];
            }
        }

        const size_t micro_fft_size = static_cast<size_t>(1) << state.micro_cfft_order;
        const size_t micro_fft_size_padded = micro_fft_size + get_transpose_padding<F>();

        F* HWY_RESTRICT micro_space0 = state.workspace.get() + 4 * state.macro_stride;
        F* HWY_RESTRICT micro_space1 = micro_space0 + 2 * state.micro_stride;

        F* HWY_RESTRICT matrix_r = state.workspace.get() + 2 * state.macro_stride;
        F* HWY_RESTRICT matrix_i = matrix_r + state.micro_segment_size * micro_fft_size_padded;
        std::complex<F>* aos_matrix = reinterpret_cast<std::complex<F>*>(matrix_r);

        // execute micro Stockham DIT CFFT
        static constexpr size_t MACRO_TILE_C = 64;
        static constexpr size_t MACRO_TILE_K = 64;
        for (size_t c_macro = 0; c_macro < state.micro_segment_size; c_macro += MACRO_TILE_C) {
            const size_t c_max = std::min(c_macro + MACRO_TILE_C, state.micro_segment_size);
            const size_t c_chunk_size = c_max - c_macro;
            // execute micro Stockham DIT CFFT (from the digital reverse block)
            for (size_t c_offset = 0; c_offset < c_chunk_size; ++c_offset) {
                const size_t reversed_c = c_macro + c_offset;
                const size_t l_idx = state.digit_rev_4[reversed_c];

                F* HWY_RESTRICT current_in = state.workspace.get() + 2 * l_idx * micro_fft_size;
                F* HWY_RESTRICT current_out = micro_space0;

                if (state.micro_stages[0] == StageType::kRadix4FirstPass) {
                    common::radix4_first_pass_aosoa(current_in, current_out, micro_fft_size);
                } else {
                    common::radix8_first_pass_aosoa(current_in, current_out, micro_fft_size);
                }

                current_in = micro_space0;
                current_out = micro_space1;
                const F* HWY_RESTRICT w_ptr = state.micro_twiddles.get();
                size_t width = (state.micro_stages[0] == StageType::kRadix4FirstPass) ? 4 : 8;
                for (size_t i = 1; i < state.micro_stages.size() - 1; ++i) {
                    const auto stage = state.micro_stages[i];
                    switch (stage) {
                    case StageType::kRadix4Width4: {
                        common::radix4_width4_aosoa(current_in, current_out, micro_fft_size, w_ptr);
                        break;
                    }
                    case StageType::kRadix4: {
                        common::radix4_aosoa(current_in, current_out, micro_fft_size, width, w_ptr);
                        break;
                    }
                    default:
                        break;
                    }
                    width = width << 2;
                    w_ptr += state.micro_twiddles_shift[i];
                    std::swap(current_in, current_out);
                }
                if constexpr (is_soa) {
                    SoAPtr<F> out_soa_ptr = make_soa<F>({
                        matrix_r + (c_macro + c_offset) * micro_fft_size_padded,
                        matrix_i + (c_macro + c_offset) * micro_fft_size_padded
                    });
                    common::radix4_last_pass_fused_aosoa<true>(current_in, out_soa_ptr, micro_fft_size, width, w_ptr);
                } else {
                    AoSPtr<F> out_aos = make_aos<F>(aos_matrix + (c_macro + c_offset) * micro_fft_size_padded);
                    common::radix4_last_pass_fused_aosoa<true>(current_in, out_aos, micro_fft_size, width, w_ptr);
                }
            }
            // perform local matrix transpose
            for (size_t k_macro = 0; k_macro < micro_fft_size; k_macro += MACRO_TILE_K) {
                const size_t k_max = std::min(k_macro + MACRO_TILE_K, micro_fft_size);
                for (size_t k = k_macro; k < k_max; ++k) {
                    const size_t out_shift = k * state.micro_segment_size + c_macro;
                    for (size_t c = 0; c < c_chunk_size; c += lanes) {
                        alignas(HWY_ALIGNMENT) F tmp_r[32];
                        alignas(HWY_ALIGNMENT) F tmp_i[32];
                        const size_t vec_len = std::min(lanes, c_chunk_size - c);
                        for (size_t i = 0; i < vec_len; ++i) {
                            if constexpr (is_soa) {
                                tmp_r[i] = matrix_r[(c_macro + c + i) * micro_fft_size_padded + k];
                                tmp_i[i] = matrix_i[(c_macro + c + i) * micro_fft_size_padded + k];
                            } else {
                                tmp_r[i] = aos_matrix[(c_macro + c + i) * micro_fft_size_padded + k].real();
                                tmp_i[i] = aos_matrix[(c_macro + c + i) * micro_fft_size_padded + k].imag();
                            }
                        }
                        auto vr = hn::Load(d, tmp_r);
                        auto vi = hn::Load(d, tmp_i);
                        common::store_complex<is_forward>(d, out_ptr.shift(OutPtr::get_complex_offset(out_shift + c)),
                                                          vr, vi);
                    }
                }
            }
        }
    }

    /**
     * dispatch RFFT pre-processing and post-processing with the correct SIMD tag
     */
    template <typename F, typename Func>
    HWY_INLINE void dispatch_rfft_tag(const size_t cfft_order, Func&& func) {
        switch (cfft_order) {
        case 0:
        case 1: {
            func(hn::CappedTag<F, 1>());
            break;
        }
        case 2: {
            func(hn::CappedTag<F, 2>());
            break;
        }
        case 3: {
            func(hn::CappedTag<F, 4>());
            break;
        }
        case 4: {
            func(hn::CappedTag<F, 8>());
            break;
        }
        default: {
            func(hn::ScalableTag<F>());
            break;
        }
        }
    }


    /**
     * execute RFFT forward post-processing, with a given SIMD tag
     * @tparam D
     * @tparam F
     * @tparam OutPtr
     * @param d
     * @param cfft_order
     * @param twiddles
     * @param in_soa
     * @param out_ptr
     */
    template <class D, typename F, typename OutPtr>
    HWY_INLINE void execute_rfft_forward_post_internal(D d, const size_t cfft_order, const F* HWY_RESTRICT twiddles,
                                                       SoAPtr<F> in_soa, OutPtr out_ptr) {
        const size_t cfft_size = 1 << cfft_order;
        static constexpr size_t lanes = hn::Lanes(d);

        F r0 = in_soa.real[0];
        F i0 = in_soa.imag[0];

        common::store_scalar<true>(out_ptr, r0 + i0, static_cast<F>(0.0));
        common::store_scalar<true>(out_ptr.shift(OutPtr::get_complex_offset(cfft_size)), r0 - i0, static_cast<F>(0.0));

        const size_t num_elements = cfft_size >> 1;
        const size_t num_blocks = (num_elements + lanes - 1) / lanes;

        for (size_t b = 0; b < num_blocks; ++b) {
            const size_t k = b * lanes + 1;

            const auto r1 = hn::LoadU(d, in_soa.real + k);
            const auto i1 = hn::LoadU(d, in_soa.imag + k);

            const auto r2_raw = hn::LoadU(d, in_soa.real + cfft_size - k - lanes + 1);
            const auto i2_raw = hn::LoadU(d, in_soa.imag + cfft_size - k - lanes + 1);

            const auto r2 = hn::Reverse(d, r2_raw);
            const auto i2 = hn::Reverse(d, i2_raw);

            const auto wc = hn::Load(d, &twiddles[b * lanes * 2]);
            const auto ws = hn::Load(d, &twiddles[b * lanes * 2 + lanes]);

            const auto si = hn::Add(i1, i2);
            const auto dr = hn::Sub(r1, r2);

            const auto xr_tmp = hn::MulAdd(si, wc, r1);
            const auto xr_k = hn::NegMulAdd(dr, ws, xr_tmp);
            const auto xi_tmp = hn::NegMulAdd(si, ws, i1);
            const auto xi_k = hn::NegMulAdd(dr, wc, xi_tmp);

            const auto xr2_tmp = hn::NegMulAdd(si, wc, r2);
            const auto xr_Mk = hn::MulAdd(dr, ws, xr2_tmp);
            const auto xi2_tmp = hn::NegMulAdd(si, ws, i2);
            const auto xi_Mk = hn::NegMulAdd(dr, wc, xi2_tmp);

            common::store_complex<true>(d, out_ptr.shift(OutPtr::get_complex_offset(k)), xr_k, xi_k);

            const auto xr_Mk_rev = hn::Reverse(d, xr_Mk);
            const auto xi_Mk_rev = hn::Reverse(d, xi_Mk);

            common::store_complex<true>(d, out_ptr.shift(OutPtr::get_complex_offset(cfft_size - k - lanes + 1)),
                                        xr_Mk_rev, xi_Mk_rev);
        }
    }

    /**
     * execute RFFT forward post-processing
     * @tparam F
     * @tparam OutPtr
     * @param cfft_order
     * @param twiddles
     * @param in_soa
     * @param out_ptr
     */
    template <typename F, typename OutPtr>
    HWY_INLINE void execute_rfft_forward_post(const size_t cfft_order, const F* HWY_RESTRICT twiddles, SoAPtr<F> in_soa,
                                              OutPtr out_ptr) {
        dispatch_rfft_tag<F>(cfft_order, [&](auto d) {
            execute_rfft_forward_post_internal(d, cfft_order, twiddles, in_soa, out_ptr);
        });
    }

    /**
     * execute RFFT backward pre-processing, with a given SIMD tag
     * @tparam D
     * @tparam F
     * @tparam InPtr
     * @param d
     * @param cfft_order
     * @param twiddles
     * @param in_ptr
     * @param out_soa
     */
    template <class D, typename F, typename InPtr>
    HWY_INLINE void execute_rfft_backward_pre_internal(
        D d,
        const size_t cfft_order,
        const F* HWY_RESTRICT twiddles,
        InPtr in_ptr, SoAPtr<F> out_soa) {
        const size_t cfft_size = 1 << cfft_order;
        static constexpr size_t lanes = hn::Lanes(d);

        F r0, i0, rM, iM;
        common::load_scalar<true>(in_ptr, r0, i0);
        common::load_scalar<true>(in_ptr.shift(InPtr::get_complex_offset(cfft_size)), rM, iM);
        out_soa.real[0] = (r0 + rM) * static_cast<F>(0.5);
        out_soa.imag[0] = (r0 - rM) * static_cast<F>(0.5);

        const size_t num_elements = cfft_size >> 1;
        const size_t num_blocks = (num_elements + lanes - 1) / lanes;

        for (size_t b = 0; b < num_blocks; ++b) {
            const size_t k = b * lanes + 1;

            hn::Vec<decltype(d)> r1, i1;
            common::load_complex<true>(d, in_ptr.shift(InPtr::get_complex_offset(k)), r1, i1);

            hn::Vec<decltype(d)> r2_raw, i2_raw;
            common::load_complex<true>(d, in_ptr.shift(InPtr::get_complex_offset(cfft_size - k - lanes + 1)), r2_raw,
                                       i2_raw);

            const auto r2 = hn::Reverse(d, r2_raw);
            const auto i2 = hn::Reverse(d, i2_raw);

            const auto wc = hn::Load(d, &twiddles[b * lanes * 2]);
            const auto ws = hn::Load(d, &twiddles[b * lanes * 2 + lanes]);

            const auto si = hn::Add(i1, i2);
            const auto dr = hn::Sub(r1, r2);

            const auto zr_tmp = hn::NegMulAdd(si, wc, r1);
            const auto zr_k = hn::NegMulAdd(dr, ws, zr_tmp);
            const auto zi_tmp = hn::NegMulAdd(si, ws, i1);
            const auto zi_k = hn::MulAdd(dr, wc, zi_tmp);

            const auto zr2_tmp = hn::MulAdd(si, wc, r2);
            const auto zr_Mk = hn::MulAdd(dr, ws, zr2_tmp);
            const auto zi2_tmp = hn::NegMulAdd(si, ws, i2);
            const auto zi_Mk = hn::MulAdd(dr, wc, zi2_tmp);

            hn::StoreU(zr_k, d, out_soa.real + k);
            hn::StoreU(zi_k, d, out_soa.imag + k);

            const auto zr_Mk_rev = hn::Reverse(d, zr_Mk);
            const auto zi_Mk_rev = hn::Reverse(d, zi_Mk);

            hn::StoreU(zr_Mk_rev, d, out_soa.real + cfft_size - k - lanes + 1);
            hn::StoreU(zi_Mk_rev, d, out_soa.imag + cfft_size - k - lanes + 1);
        }
    }

    /**
     * execute RFFT backward pre-processing
     * @tparam F
     * @tparam InPtr
     * @param cfft_order
     * @param twiddles
     * @param in_ptr
     * @param out_soa
     */
    template <typename F, typename InPtr>
    HWY_INLINE void execute_rfft_backward_pre(const size_t cfft_order, const F* HWY_RESTRICT twiddles, InPtr in_ptr,
                                              SoAPtr<F> out_soa) {
        dispatch_rfft_tag<F>(cfft_order, [&](auto d) {
            execute_rfft_backward_pre_internal(d, cfft_order, twiddles, in_ptr, out_soa);
        });
    }

    /**
     * execute RFFT forward post-processing and compute squared magnitude, with a given SIMD tag
     * @tparam D
     * @tparam F
     * @param d
     * @param cfft_order
     * @param twiddles
     * @param in_soa
     * @param out_ptr
     */
    template <class D, typename F>
    HWY_INLINE void execute_rfft_forward_sqr_mag_post_internal(D d, const size_t cfft_order,
                                                               const F* HWY_RESTRICT twiddles, SoAPtr<F> in_soa,
                                                               F* HWY_RESTRICT out_ptr) {

        const size_t cfft_size = 1 << cfft_order;
        static constexpr size_t lanes = hn::Lanes(d);

        F r0 = in_soa.real[0];
        F i0 = in_soa.imag[0];

        const F dc = r0 + i0;
        const F nyquist = r0 - i0;

        out_ptr[0] = dc * dc;
        out_ptr[cfft_size] = nyquist * nyquist;

        const size_t num_elements = cfft_size >> 1;
        const size_t num_blocks = (num_elements + lanes - 1) / lanes;

        for (size_t b = 0; b < num_blocks; ++b) {
            const size_t k = b * lanes + 1;

            const auto r1 = hn::LoadU(d, in_soa.real + k);
            const auto i1 = hn::LoadU(d, in_soa.imag + k);

            const auto r2_raw = hn::LoadU(d, in_soa.real + cfft_size - k - lanes + 1);
            const auto i2_raw = hn::LoadU(d, in_soa.imag + cfft_size - k - lanes + 1);

            const auto r2 = hn::Reverse(d, r2_raw);
            const auto i2 = hn::Reverse(d, i2_raw);

            const auto wc = hn::Load(d, &twiddles[b * lanes * 2]);
            const auto ws = hn::Load(d, &twiddles[b * lanes * 2 + lanes]);

            const auto si = hn::Add(i1, i2);
            const auto dr = hn::Sub(r1, r2);

            const auto xr_tmp = hn::MulAdd(si, wc, r1);
            const auto xr_k = hn::NegMulAdd(dr, ws, xr_tmp);
            const auto xi_tmp = hn::NegMulAdd(si, ws, i1);
            const auto xi_k = hn::NegMulAdd(dr, wc, xi_tmp);

            const auto xr2_tmp = hn::NegMulAdd(si, wc, r2);
            const auto xr_Mk = hn::MulAdd(dr, ws, xr2_tmp);
            const auto xi2_tmp = hn::NegMulAdd(si, ws, i2);
            const auto xi_Mk = hn::NegMulAdd(dr, wc, xi2_tmp);

            const auto mag2_k = hn::MulAdd(xr_k, xr_k, hn::Mul(xi_k, xi_k));
            const auto mag2_Mk = hn::MulAdd(xr_Mk, xr_Mk, hn::Mul(xi_Mk, xi_Mk));

            hn::StoreU(mag2_k, d, out_ptr + k);

            const auto mag2_Mk_rev = hn::Reverse(d, mag2_Mk);
            hn::StoreU(mag2_Mk_rev, d, out_ptr + cfft_size - k - lanes + 1);
        }
    }

    /**
     * execute RFFT forward post-processing and compute squared magnitude
     * @tparam F
     * @param cfft_order
     * @param twiddles
     * @param in_soa
     * @param out_ptr
     */
    template <typename F>
    HWY_INLINE void execute_rfft_forward_sqr_mag_post(const size_t cfft_order, const F* HWY_RESTRICT twiddles,
                                                      SoAPtr<F> in_soa, F* HWY_RESTRICT out_ptr) {
        dispatch_rfft_tag<F>(cfft_order, [&](auto d) {
            execute_rfft_forward_sqr_mag_post_internal(d, cfft_order, twiddles, in_soa, out_ptr);
        });
    }
}
