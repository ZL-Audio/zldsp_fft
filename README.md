# zldsp::fft

A header-only C++ Fast Fourier Transform (FFT) library built with [Google Highway](https://github.com/google/highway).

If you want to develop zldsp::fft, please refer to [zldsp_fft_develop](https://github.com/ZL-Audio/zldsp_fft_develop).

## Key Features

- Header-Only: This library does not require any CMake target or external binary.
- Cross-Platform SIMD: This library supports SSE2/SSE4/AVX2/NEON.
- Flexible Data Layouts: This library supports AoS/SoA for complex numbers.

## Requirements

1. C++ Standard: C++20 or higher
2. Google Highway: You must include and link Google Highway in your project. The headers in this library expect the following includes to be resolvable:
```cpp
#include <hwy/aligned_allocator.h>
#include <hwy/highway.h>
```

## Usage

### Static Dispatch

```cmake
target_compile_definitions(my_static_target PRIVATE HWY_COMPILE_ONLY_STATIC)
```

Use the compiler's architecture option to select an SSE, AVX2, or NEON static
target.

| SIMD Target | GCC/Clang                         | MSVC             |
| ----------- | --------------------------------- | ---------------- |
| SSE2        | `-march=x86-64`                   | no flag required |
| SSE4        | `-march=x86-64-v2 -maes -mpclmul` | not supported    |
| AVX2        | `-march=x86-64-v3 -maes -mpclmul` | `/arch:AVX2`     |
| NEON        | `-march=armv8-a+simd`             | `/arch:armv8.0`  |

See [`static_dispatch_caller`](https://github.com/ZL-Audio/zldsp_fft_develop/tree/main/examples/static_dispatch_caller.cpp) for CFFT and RFFT static dispatch examples with AoS and SoA layouts.

### Caller-owned Dynamic Dispatch

For example, an x86 application can enable only SSE2 and AVX2:

```cmake
target_compile_definitions(my_dynamic_target PRIVATE "HWY_DISABLED_TARGETS=~(HWY_SSE2|HWY_AVX2)")
```

Compile this target for the oldest supported baseline (for example, `-march=x86-64` for SSE2).

See [`wrapper`](https://github.com/ZL-Audio/zldsp_fft_develop/tree/main/examples/dynamic_dispatch_wrapper.cpp), its [`interface`](https://github.com/ZL-Audio/zldsp_fft_develop/tree/main/examples/dynamic_dispatch_wrapper.hpp), and [`dynamic_dispatch_caller`](https://github.com/ZL-Audio/zldsp_fft_develop/tree/main/examples/dynamic_dispatch_caller.cpp) for CFFT and RFFT dynamic dispatch examples with AoS and SoA layouts.

## License

zldsp::fft is licensed under Apache-2.0 license, as found in the [LICENSE.md](LICENSE.md) file.

## Reference:

- Van Loan, Charles. Computational frameworks for the fast Fourier transform. Society for Industrial and Applied Mathematics, 1992.
- [Notes on FFTs: for implementers](https://fgiesen.wordpress.com/2023/03/19/notes-on-ffts-for-implementers/)
- [OTFFT documentation](http://wwwa.pikara.ne.jp/okojisan/otfft-en/stockham1.html)