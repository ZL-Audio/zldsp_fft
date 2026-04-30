# zldsp::fft

A header-only C++ Fast Fourier Transform (FFT) library built with [Google Highway](https://github.com/google/highway).

## Key Features

- Header-Only: This library does not require any CMake target or external binary.
- Cross-Platform SIMD: This library supports SSE2/SSE4/AVX2/NEON.
- Flexible Data Layouts: This library supports AoS/SoA for complex numbers

## Requirements

1. C++ Standard: C++20 or higher
2. Google Highway: You must include and link Google Highway in your project. The headers in this library expect the following includes to be resolvable:
```cpp
#include <hwy/aligned_allocator.h>
#include <hwy/highway.h>
```

## Integration & Compiler Flags

To activate SIMD, you must set the correct compilation flags. Highway will automatically detect the target architecture from these flags.

### SSE2

- GCC/Clang: `-march=x86-64`
- MSVC: no flag required

### AVX2

- GCC/Clang: `-march=x86-64-v3 -maes -mpclmul`
- MSVC: `/arch:AVX2`

### NEON

- GCC/Clang: `-march=armv8-a+simd`

## Usage

Include the relevant headers from the `/src` directory.

Both `CFFT` and `RFFT` are templated on the floating-point type (e.g., `float` or `double`) and are instantiated using the base-2 **order** of the FFT (where `size = 1 << order`).

### Complex FFT (`CFFT`)

```cpp
#include "src/zldsp_fft_cfft.hpp"
#include <vector>
#include <complex>

// a CFFT of size 1024 (2^10) using float
constexpr size_t order = 10; 
zldsp::fft::CFFT<float> cfft(order);

std::vector<std::complex<float>> in_buffer(1 << order);
std::vector<std::complex<float>> out_buffer(1 << order);

// forward transform (AoS to AoS)
cfft.forward(in_buffer.data(), out_buffer.data());

// backward transform (AoS to AoS)
cfft.backward(out_buffer.data(), in_buffer.data());
```

### Real FFT (`RFFT`)

```cpp
#include "src/zldsp_fft_rfft.hpp"
#include <vector>
#include <complex>

// a RFFT of size 1024 (2^10) using float
constexpr size_t order = 10;
zldsp::fft::RFFT<float> rfft(order);

std::vector<float> real_in(1 << order);
std::vector<std::complex<float>> complex_out((1 << order) / 2 + 1);

// forward transform (real to AoS)
rfft.forward(real_in.data(), complex_out.data());

// backward transform (AoS to real)
rfft.backward(complex_out.data(), real_in.data());
```

### Data Layouts (AoS vs. SoA)

Both `CFFT` and `RFFT` support AoS/SoA for complex numbers:

AoS stores interleaved real/imaginary values in one array:
```cpp
std::vector<std::complex<float>> out_buffer(1 << order);
```

SoA stores continuous real/imaginary values in two arrays:

```cpp
std::vector<float> out_real(1 << order);
std::vector<float> out_imag(1 << order);
```

Example using SoA:
```cpp
#include "src/zldsp_fft_cfft.hpp"
#include <vector>
#include <complex>

// a CFFT of size 1024 (2^10) using float
constexpr size_t order = 10; 
zldsp::fft::CFFT<float> cfft(order);

std::vector<std::complex<float>> in_buffer(1 << order);
std::vector<float> out_real(1 << order);
std::vector<float> out_imag(1 << order);

// forward transform (AoS to SoA)
cfft.forward(in_buffer.data(), {out_real.data(), out_imag.data()});

// backward transform (SoA to AoS)
cfft.backward({out_real.data(), out_imag.data()}, in_buffer.data());
```

## License

zldsp::fft is licensed under Apache-2.0 license, as found in the [LICENSE.md](LICENSE.md) file.

## Reference:

- Van Loan, Charles. Computational frameworks for the fast Fourier transform. Society for Industrial and Applied Mathematics, 1992.
- [Notes on FFTs: for implementers](https://fgiesen.wordpress.com/2023/03/19/notes-on-ffts-for-implementers/)
- [OTFFT documentation](http://wwwa.pikara.ne.jp/okojisan/otfft-en/stockham1.html)