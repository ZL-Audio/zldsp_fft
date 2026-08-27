#pragma once

#include "zldsp_fft_common_init.hpp"
#include "zldsp_fft_common_radix.hpp"
#include "zldsp_fft_common_kernel.hpp"

namespace zldsp::fft::common {
    namespace hn = hwy::HWY_NAMESPACE;

    template <typename F, size_t width>
    inline constexpr size_t fixed_stockham_twiddle_stride() noexcept {
        constexpr size_t lanes = hn::MaxLanes(hn::ScalableTag<F>());
        if constexpr (width == 4) {
            return 6 * std::max<size_t>(4, lanes);
        } else {
            return 6 * std::max<size_t>(width, lanes);
        }
    }

    template <size_t n, size_t width, bool is_forward,
              typename F, typename OutPtr>
    HWY_INLINE void execute_stockham_fixed_tail(
        F* HWY_RESTRICT current_in, F* HWY_RESTRICT current_out,
        const F* HWY_RESTRICT w_ptr, OutPtr out_ptr) noexcept {
        static_assert(width <= n / 4);
        if constexpr (width == n / 4) {
            common::radix4_last_pass_fused_aosoa<is_forward>(
                current_in, out_ptr, n, width, w_ptr);
        } else {
            common::radix4_aosoa(current_in, current_out, n, width, w_ptr);
            execute_stockham_fixed_tail<n, width * 4, is_forward>(
                current_out, current_in,
                w_ptr + fixed_stockham_twiddle_stride<F, width>(), out_ptr);
        }
    }

    template <size_t order, bool is_forward,
              typename F, typename InPtr, typename OutPtr>
    HWY_INLINE void execute_stockham_cfft_fixed(
        F* HWY_RESTRICT workspace, const F* HWY_RESTRICT twiddles,
        InPtr in_ptr, OutPtr out_ptr) noexcept {
        static_assert(order >= 6);
        static constexpr size_t n = static_cast<size_t>(1) << order;
        static constexpr size_t stride = get_cfft_stride<F>(n);

        F* HWY_RESTRICT buffer0 = workspace;
        F* HWY_RESTRICT buffer1 = workspace + 2 * stride;

        if constexpr ((order & 1) == 0) {
            common::radix4_first_pass_fused_aosoa<is_forward>(
                in_ptr, buffer1, n);
            common::radix4_width4_aosoa(buffer1, buffer0, n, twiddles);
            execute_stockham_fixed_tail<n, 16, is_forward>(
                buffer0, buffer1,
                twiddles + fixed_stockham_twiddle_stride<F, 4>(), out_ptr);
        } else {
            common::radix8_first_pass_fused_aosoa<is_forward>(
                in_ptr, buffer1, n);
            common::radix4_aosoa(buffer1, buffer0, n, 8, twiddles);
            execute_stockham_fixed_tail<n, 32, is_forward>(
                buffer0, buffer1,
                twiddles + fixed_stockham_twiddle_stride<F, 8>(), out_ptr);
        }
    }

    template <size_t order, typename F>
    HWY_INLINE void execute_stockham_micro_fixed(
        const CFFTState<F>& state,
        F* HWY_RESTRICT macro_leaf, F* HWY_RESTRICT micro_workspace,
        SoAPtr<F> out_ptr) noexcept {
        static_assert(order >= 6);
        static constexpr size_t n = static_cast<size_t>(1) << order;
        const F* HWY_RESTRICT twiddles = state.micro_twiddles.get();

        if constexpr ((order & 1) == 0) {
            common::radix4_first_pass_aosoa(
                macro_leaf, micro_workspace, n);
            common::radix4_width4_aosoa(
                micro_workspace, macro_leaf, n, twiddles);
            execute_stockham_fixed_tail<n, 16, true>(
                macro_leaf, micro_workspace,
                twiddles + fixed_stockham_twiddle_stride<F, 4>(), out_ptr);
        } else {
            common::radix8_first_pass_aosoa(
                macro_leaf, micro_workspace, n);
            common::radix4_aosoa(
                micro_workspace, macro_leaf, n, 8, twiddles);
            execute_stockham_fixed_tail<n, 32, true>(
                macro_leaf, micro_workspace,
                twiddles + fixed_stockham_twiddle_stride<F, 8>(), out_ptr);
        }
    }

    template <typename F>
    HWY_INLINE void execute_stockham_micro_generic(
        const CFFTState<F>& state,
        F* HWY_RESTRICT macro_leaf, F* HWY_RESTRICT micro_workspace,
        SoAPtr<F> out_ptr) noexcept {
        const size_t micro_fft_size =
            static_cast<size_t>(1) << state.micro_cfft_order;
        F* HWY_RESTRICT current_in = macro_leaf;
        F* HWY_RESTRICT current_out = micro_workspace;

        if (state.micro_stages[0] == StageType::kRadix4FirstPass) {
            common::radix4_first_pass_aosoa(
                current_in, current_out, micro_fft_size);
        } else {
            common::radix8_first_pass_aosoa(
                current_in, current_out, micro_fft_size);
        }

        current_in = micro_workspace;
        current_out = macro_leaf;
        const F* HWY_RESTRICT w_ptr = state.micro_twiddles.get();
        size_t width =
            (state.micro_stages[0] == StageType::kRadix4FirstPass) ? 4 : 8;
        {
            if (state.micro_stages[1] == StageType::kRadix4Width4) {
                common::radix4_width4_aosoa(
                    current_in, current_out, micro_fft_size, w_ptr);
            } else {
                common::radix4_aosoa(
                    current_in, current_out, micro_fft_size, width, w_ptr);
            }
            width = width << 2;
            w_ptr += state.micro_twiddle_strides[1];
            std::swap(current_in, current_out);
        }
        for (size_t i = 2; i < state.micro_stages.size() - 1; ++i) {
            common::radix4_aosoa(
                current_in, current_out, micro_fft_size, width, w_ptr);
            width = width << 2;
            w_ptr += state.micro_twiddle_strides[i];
            std::swap(current_in, current_out);
        }
        common::radix4_last_pass_fused_aosoa<true>(
            current_in, out_ptr, micro_fft_size, width, w_ptr);
    }

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
     * @param twiddle_strides
     * @param stages
     * @param in_ptr
     * @param out_ptr
     */
    template <bool is_forward, typename F, typename InPtr, typename OutPtr>
    HWY_INLINE void execute_stockham_cfft(const size_t cfft_order, const size_t stride, F* HWY_RESTRICT workspace,
                                          const F* HWY_RESTRICT twiddles, const std::vector<size_t>& twiddle_strides,
                                          const std::vector<StageType>& stages,
                                          InPtr in_ptr, OutPtr out_ptr) noexcept {
        const size_t cfft_size = 1 << cfft_order;
        switch (cfft_order) {
        case 0: {
            common::execute_cfft_order_0<is_forward, F>(in_ptr, out_ptr);
            return;
        }
        case 1: {
            common::execute_cfft_order_1<is_forward, F>(in_ptr, out_ptr);
            return;
        }
        case 2: {
            common::execute_cfft_order_2<is_forward, F>(in_ptr, out_ptr);
            return;
        }
        case 3: {
            common::execute_cfft_order_3<is_forward, F>(in_ptr, out_ptr);
            return;
        }
        case 4: {
            common::execute_cfft_order_4<is_forward, F>(in_ptr, out_ptr, twiddles, twiddles + 60);
            return;
        }
        case 5: {
            common::execute_cfft_order_5<is_forward, F>(in_ptr, out_ptr, twiddles, twiddles + 120);
            return;
        }
        case 6:
            execute_stockham_cfft_fixed<6, is_forward>(workspace, twiddles, in_ptr, out_ptr);
            return;
        case 7:
            execute_stockham_cfft_fixed<7, is_forward>(workspace, twiddles, in_ptr, out_ptr);
            return;
        case 8:
            execute_stockham_cfft_fixed<8, is_forward>(workspace, twiddles, in_ptr, out_ptr);
            return;
        case 9:
            execute_stockham_cfft_fixed<9, is_forward>(workspace, twiddles, in_ptr, out_ptr);
            return;
        case 10:
            execute_stockham_cfft_fixed<10, is_forward>(workspace, twiddles, in_ptr, out_ptr);
            return;
        case 11:
            execute_stockham_cfft_fixed<11, is_forward>(workspace, twiddles, in_ptr, out_ptr);
            return;
        case 12:
            execute_stockham_cfft_fixed<12, is_forward>(workspace, twiddles, in_ptr, out_ptr);
            return;
        case 13:
            execute_stockham_cfft_fixed<13, is_forward>(workspace, twiddles, in_ptr, out_ptr);
            return;
        case 14:
            execute_stockham_cfft_fixed<14, is_forward>(workspace, twiddles, in_ptr, out_ptr);
            return;
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
        size_t width = (stages[0] == StageType::kRadix4FirstPass) ? 4 : 8;
        {
            if (stages[1] == StageType::kRadix4Width4) {
                common::radix4_width4_aosoa(out_aosoa, in_aosoa, cfft_size, w_ptr);
            } else {
                common::radix4_aosoa(out_aosoa, in_aosoa, cfft_size, width, w_ptr);
            }
            width = width << 2;
            w_ptr += twiddle_strides[1];
        }
        for (size_t i = 2; i < stages.size() - 1; ++i) {
            common::radix4_aosoa(in_aosoa, out_aosoa, cfft_size, width, w_ptr);
            width = width << 2;
            w_ptr += twiddle_strides[i];
            std::swap(in_aosoa, out_aosoa);
        }
        common::radix4_last_pass_fused_aosoa<is_forward>(in_aosoa, out_ptr, cfft_size, width, w_ptr);
    }

    /**
     * transpose one SIMD-lane-square tile and store it in column-major order
     * @tparam D
     * @tparam F
     * @param d
     * @param in
     * @param in_stride
     * @param out
     * @param out_stride
     */
    template <class D, typename F>
    HWY_INLINE void transpose_square_component_to_soa(
        D d, const F* HWY_RESTRICT in, const size_t in_stride,
        F* HWY_RESTRICT out, const size_t out_stride) noexcept {
        static constexpr size_t lanes = hn::MaxLanes(d);

        if constexpr (lanes == 2) {
            const auto v0 = hn::Load(d, in);
            const auto v1 = hn::Load(d, in + in_stride);
            hn::StoreU(hn::InterleaveLower(d, v0, v1), d, out);
            hn::StoreU(hn::InterleaveUpper(d, v0, v1), d, out + out_stride);
        } else if constexpr (lanes == 4) {
            const auto v0 = hn::Load(d, in);
            const auto v1 = hn::Load(d, in + in_stride);
            const auto v2 = hn::Load(d, in + 2 * in_stride);
            const auto v3 = hn::Load(d, in + 3 * in_stride);

            hn::Vec<decltype(d)> o0, o1, o2, o3;
            transpose4x4(d, v0, v1, v2, v3, o0, o1, o2, o3);
            hn::StoreU(o0, d, out);
            hn::StoreU(o1, d, out + out_stride);
            hn::StoreU(o2, d, out + 2 * out_stride);
            hn::StoreU(o3, d, out + 3 * out_stride);
        } else if constexpr (lanes == 8) {
            using DH = hn::Half<D>;
            const DH dh;
            HWY_UNROLL(1)
            for (size_t col = 0; col < 8; col += 4) {
                const auto v0 = hn::Load(dh, in + col);
                const auto v1 = hn::Load(dh, in + in_stride + col);
                const auto v2 = hn::Load(dh, in + 2 * in_stride + col);
                const auto v3 = hn::Load(dh, in + 3 * in_stride + col);
                hn::Vec<DH> lo0, lo1, lo2, lo3;
                transpose4x4(dh, v0, v1, v2, v3, lo0, lo1, lo2, lo3);

                const auto v4 = hn::Load(dh, in + 4 * in_stride + col);
                const auto v5 = hn::Load(dh, in + 5 * in_stride + col);
                const auto v6 = hn::Load(dh, in + 6 * in_stride + col);
                const auto v7 = hn::Load(dh, in + 7 * in_stride + col);
                hn::Vec<DH> hi0, hi1, hi2, hi3;
                transpose4x4(dh, v4, v5, v6, v7, hi0, hi1, hi2, hi3);

                hn::StoreU(hn::Combine(d, hi0, lo0), d, out + col * out_stride);
                hn::StoreU(hn::Combine(d, hi1, lo1), d, out + (col + 1) * out_stride);
                hn::StoreU(hn::Combine(d, hi2, lo2), d, out + (col + 2) * out_stride);
                hn::StoreU(hn::Combine(d, hi3, lo3), d, out + (col + 3) * out_stride);
            }
        }
    }

    /**
     * transpose a 4x4 real/imaginary tile and immediately write AoS columns
     * @tparam is_forward
     * @tparam D
     * @tparam F
     * @tparam OutPtr
     * @param d
     * @param in_r
     * @param in_i
     * @param in_stride
     * @param out
     * @param out_stride
     */
    template <bool is_forward, class D, typename F, typename OutPtr>
    HWY_INLINE void transpose_store_4x4_aos(
        D d, const F* HWY_RESTRICT in_r, const F* HWY_RESTRICT in_i,
        const size_t in_stride, OutPtr out, const size_t out_stride) noexcept {
        const auto rr0 = hn::Load(d, in_r);
        const auto rr1 = hn::Load(d, in_r + in_stride);
        const auto rr2 = hn::Load(d, in_r + 2 * in_stride);
        const auto rr3 = hn::Load(d, in_r + 3 * in_stride);
        hn::Vec<D> r0, r1, r2, r3;
        transpose4x4(d, rr0, rr1, rr2, rr3, r0, r1, r2, r3);

        const auto ii0 = hn::Load(d, in_i);
        const auto ii1 = hn::Load(d, in_i + in_stride);
        const auto ii2 = hn::Load(d, in_i + 2 * in_stride);
        const auto ii3 = hn::Load(d, in_i + 3 * in_stride);
        hn::Vec<D> i0, i1, i2, i3;
        transpose4x4(d, ii0, ii1, ii2, ii3, i0, i1, i2, i3);

        common::store_complex<is_forward>(d, out, r0, i0);
        common::store_complex<is_forward>(
            d, out.shift(OutPtr::get_complex_offset(out_stride)), r1, i1);
        common::store_complex<is_forward>(
            d, out.shift(OutPtr::get_complex_offset(2 * out_stride)), r2, i2);
        common::store_complex<is_forward>(
            d, out.shift(OutPtr::get_complex_offset(3 * out_stride)), r3, i3);
    }

    /**
     * transpose a 2x2 real/imaginary tile and immediately write AoS columns
     * @tparam is_forward
     * @tparam D
     * @tparam F
     * @tparam OutPtr
     * @param d
     * @param in_r
     * @param in_i
     * @param in_stride
     * @param out
     * @param out_stride
     */
    template <bool is_forward, class D, typename F, typename OutPtr>
    HWY_INLINE void transpose_store_2x2_aos(
        D d, const F* HWY_RESTRICT in_r, const F* HWY_RESTRICT in_i,
        const size_t in_stride, OutPtr out, const size_t out_stride) noexcept {
        const auto rr0 = hn::Load(d, in_r);
        const auto rr1 = hn::Load(d, in_r + in_stride);
        const auto r0 = hn::InterleaveLower(d, rr0, rr1);
        const auto r1 = hn::InterleaveUpper(d, rr0, rr1);

        const auto ii0 = hn::Load(d, in_i);
        const auto ii1 = hn::Load(d, in_i + in_stride);
        const auto i0 = hn::InterleaveLower(d, ii0, ii1);
        const auto i1 = hn::InterleaveUpper(d, ii0, ii1);

        common::store_complex<is_forward>(d, out, r0, i0);
        common::store_complex<is_forward>(
            d, out.shift(OutPtr::get_complex_offset(out_stride)), r1, i1);
    }

    template <bool is_forward, typename F, typename OutPtr>
    HWY_NOINLINE void transpose_hybrid_tile(
        const CFFTState<F>& state,
        const F* HWY_RESTRICT transpose_tile_r, const F* HWY_RESTRICT transpose_tile_i,
        const size_t micro_fft_size, const size_t micro_fft_size_padded,
        const size_t tile_row_begin, const size_t num_tile_rows,
        OutPtr out_ptr) noexcept {
        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::MaxLanes(d);
        static constexpr bool is_soa = std::is_same_v<OutPtr, SoAPtr<F>>;

        HWY_ASSUME(num_tile_rows >= lanes);
        HWY_ASSUME((num_tile_rows % lanes) == 0);
        HWY_ASSUME(micro_fft_size >= lanes);
        HWY_ASSUME((micro_fft_size % lanes) == 0);
        for (size_t tile_col = 0; tile_col < micro_fft_size; tile_col += lanes) {
            for (size_t tile_row = 0; tile_row < num_tile_rows; tile_row += lanes) {
                const size_t tile_offset = tile_row * micro_fft_size_padded + tile_col;
                const size_t output_offset = tile_col * state.num_micro_ffts + tile_row_begin + tile_row;

                if constexpr (is_soa && (lanes == 2 || lanes == 4 || lanes == 8)) {
                    F* HWY_RESTRICT out_r = is_forward ? out_ptr.real : out_ptr.imag;
                    F* HWY_RESTRICT out_i = is_forward ? out_ptr.imag : out_ptr.real;
                    common::transpose_square_component_to_soa(
                        d, transpose_tile_r + tile_offset,
                        micro_fft_size_padded,
                        out_r + output_offset, state.num_micro_ffts);
                    common::transpose_square_component_to_soa(
                        d, transpose_tile_i + tile_offset,
                        micro_fft_size_padded,
                        out_i + output_offset, state.num_micro_ffts);
                } else if constexpr (!is_soa && lanes == 8) {
                    using DH = hn::Half<decltype(d)>;
                    const DH dh;
                    HWY_UNROLL(1)
                    for (size_t subcolumn = 0; subcolumn < 8; subcolumn += 4) {
                        const size_t column_output_offset = output_offset + subcolumn * state.num_micro_ffts;
                        common::transpose_store_4x4_aos<is_forward>(
                            dh, transpose_tile_r + tile_offset + subcolumn,
                            transpose_tile_i + tile_offset + subcolumn,
                            micro_fft_size_padded,
                            out_ptr.shift(OutPtr::get_complex_offset(
                                column_output_offset)),
                            state.num_micro_ffts);
                        common::transpose_store_4x4_aos<is_forward>(
                            dh, transpose_tile_r + tile_offset + subcolumn +
                            4 * micro_fft_size_padded,
                            transpose_tile_i + tile_offset + subcolumn +
                            4 * micro_fft_size_padded,
                            micro_fft_size_padded,
                            out_ptr.shift(OutPtr::get_complex_offset(
                                column_output_offset + 4)),
                            state.num_micro_ffts);
                    }
                } else if constexpr (!is_soa && lanes == 4) {
                    common::transpose_store_4x4_aos<is_forward>(
                        d, transpose_tile_r + tile_offset,
                        transpose_tile_i + tile_offset,
                        micro_fft_size_padded,
                        out_ptr.shift(OutPtr::get_complex_offset(output_offset)),
                        state.num_micro_ffts);
                } else if constexpr (!is_soa && lanes == 2) {
                    common::transpose_store_2x2_aos<is_forward>(
                        d, transpose_tile_r + tile_offset,
                        transpose_tile_i + tile_offset,
                        micro_fft_size_padded,
                        out_ptr.shift(OutPtr::get_complex_offset(output_offset)),
                        state.num_micro_ffts);
                } else {
                    alignas(HWY_ALIGNMENT) F tmp_r[32];
                    alignas(HWY_ALIGNMENT) F tmp_i[32];
                    for (size_t kk = 0; kk < lanes; ++kk) {
                        for (size_t i = 0; i < lanes; ++i) {
                            tmp_r[i] = transpose_tile_r[(tile_row + i) * micro_fft_size_padded + tile_col + kk];
                            tmp_i[i] = transpose_tile_i[(tile_row + i) * micro_fft_size_padded + tile_col + kk];
                        }
                        const auto vr = hn::Load(d, tmp_r);
                        const auto vi = hn::Load(d, tmp_i);
                        common::store_complex<is_forward>(
                            d, out_ptr.shift(OutPtr::get_complex_offset(
                                output_offset + kk * state.num_micro_ffts)), vr, vi);
                    }
                }
            }
        }
    }

    template <size_t micro_order, bool is_forward, typename F, typename OutPtr>
    HWY_NOINLINE void execute_hybrid_tail(
        const CFFTState<F>& state, OutPtr out_ptr) noexcept {
        static_assert(micro_order == 0 || (micro_order >= 8 && micro_order <= 10));
        const size_t micro_fft_size = [&]() {
            if constexpr (micro_order == 0) {
                return static_cast<size_t>(1) << state.micro_cfft_order;
            } else {
                return static_cast<size_t>(1) << micro_order;
            }
        }();
        const size_t micro_fft_size_padded = micro_fft_size + get_transpose_padding<F>();

        static constexpr size_t kTransposeTileRows = kHybridTransposeTileRows;
        F* HWY_RESTRICT transpose_tile_r =
            state.workspace.get() + 2 * state.macro_stride;
        F* HWY_RESTRICT transpose_tile_i =
            transpose_tile_r + state.transpose_tile_stride;
        F* HWY_RESTRICT micro_workspace =
            transpose_tile_i + state.transpose_tile_stride;

        // process micro CFFTs one reusable transpose tile at a time
        for (size_t tile_row_begin = 0; tile_row_begin < state.num_micro_ffts; tile_row_begin += kTransposeTileRows) {
            const size_t tile_row_end = std::min<size_t>(tile_row_begin + kTransposeTileRows, state.num_micro_ffts);
            const size_t num_tile_rows = tile_row_end - tile_row_begin;

            // execute micro CFFTs in natural output-row order
            for (size_t tile_row = 0; tile_row < num_tile_rows; ++tile_row) {
                const size_t output_row = tile_row_begin + tile_row;
                const size_t macro_leaf_index = state.radix4_digit_reversal[output_row];

                F* HWY_RESTRICT macro_leaf = state.workspace.get() + 2 * macro_leaf_index * micro_fft_size;
                SoAPtr<F> tile_row_output = make_soa<F>({
                    transpose_tile_r + tile_row * micro_fft_size_padded,
                    transpose_tile_i + tile_row * micro_fft_size_padded
                });
                if constexpr (micro_order == 0) {
                    execute_stockham_micro_generic(
                        state, macro_leaf, micro_workspace, tile_row_output);
                } else {
                    execute_stockham_micro_fixed<micro_order>(
                        state, macro_leaf, micro_workspace, tile_row_output);
                }
            }

            transpose_hybrid_tile<is_forward>(
                state, transpose_tile_r, transpose_tile_i,
                micro_fft_size, micro_fft_size_padded,
                tile_row_begin, num_tile_rows, out_ptr);
        }
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
    HWY_INLINE void execute_cfft(const CFFTState<F>& state, InPtr in_ptr, OutPtr out_ptr) noexcept {
        // small FFT, dispatch to pure Stockham CFFT
        if (state.num_macro_stages == 0) {
            execute_stockham_cfft<is_forward, F, InPtr, OutPtr>(
                state.micro_cfft_order,
                state.micro_stride,
                state.workspace.get(),
                state.micro_twiddles.get(),
                state.micro_twiddle_strides,
                state.micro_stages,
                in_ptr, out_ptr);
            return;
        }
        // execute macro Cooley-Tukey DIF CFFT
        {
            F* HWY_RESTRICT macro_space = state.workspace.get();
            const F* HWY_RESTRICT w_ptr = state.macro_twiddles.get();
            common::radix4_first_pass_dif_fused_aosoa<is_forward>(in_ptr, macro_space, state.cfft_size, w_ptr);
            w_ptr += state.macro_twiddle_strides[0];

            const size_t remaining_stages = state.num_macro_stages - 1;
            if (remaining_stages != 0) {
                const size_t width = state.cfft_size >> 4;
                common::radix4_dif_aosoa_depth_first(
                    macro_space, state.cfft_size, width, w_ptr,
                    state.macro_twiddle_strides.data() + 1,
                    remaining_stages);
            }
        }

        // dispatch one fixed micro order per transform
        switch (state.micro_cfft_order) {
        case 8:
            execute_hybrid_tail<8, is_forward>(state, out_ptr);
            return;
        case 9:
            execute_hybrid_tail<9, is_forward>(state, out_ptr);
            return;
        case 10:
            execute_hybrid_tail<10, is_forward>(state, out_ptr);
            return;
        default:
            execute_hybrid_tail<0, is_forward>(state, out_ptr);
            return;
        }
    }

    /**
     * dispatch RFFT pre-processing and post-processing with the correct SIMD tag
     * @tparam F
     * @tparam Func
     * @param cfft_order
     * @param func
     */
    template <typename F, typename Func>
    HWY_INLINE void dispatch_rfft_tag(const size_t cfft_order, Func&& func) noexcept {
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
                                                       SoAPtr<F> in_soa, OutPtr out_ptr) noexcept {
        const size_t cfft_size = 1 << cfft_order;
        static constexpr size_t lanes = hn::MaxLanes(d);

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
    HWY_INLINE void execute_rfft_forward_post(const size_t cfft_order, const F* HWY_RESTRICT twiddles,
                                              SoAPtr<F> in_soa, OutPtr out_ptr) noexcept {
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
    HWY_INLINE void execute_rfft_backward_pre_internal(D d, const size_t cfft_order, const F* HWY_RESTRICT twiddles,
                                                       InPtr in_ptr, SoAPtr<F> out_soa) noexcept {
        const size_t cfft_size = 1 << cfft_order;
        static constexpr size_t lanes = hn::MaxLanes(d);

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
    HWY_INLINE void execute_rfft_backward_pre(const size_t cfft_order, const F* HWY_RESTRICT twiddles,
                                              InPtr in_ptr, SoAPtr<F> out_soa) noexcept {
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
                                                               const F* HWY_RESTRICT twiddles,
                                                               SoAPtr<F> in_soa, F* HWY_RESTRICT out_ptr) noexcept {
        const size_t cfft_size = 1 << cfft_order;
        static constexpr size_t lanes = hn::MaxLanes(d);

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
                                                      SoAPtr<F> in_soa, F* HWY_RESTRICT out_ptr) noexcept {
        dispatch_rfft_tag<F>(cfft_order, [&](auto d) {
            execute_rfft_forward_sqr_mag_post_internal(d, cfft_order, twiddles, in_soa, out_ptr);
        });
    }
}
