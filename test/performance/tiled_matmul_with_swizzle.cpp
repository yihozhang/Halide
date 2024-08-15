#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_test_dirs.h"

#include <iomanip>
#include <iostream>
// #define ON_X86
#ifdef ON_X86
#include <x86intrin.h>
#endif

using namespace Halide;

template<typename IntT>
void fill_buffer(Buffer<IntT> &buf, int col, int row) {
    for (int iy = 0; iy < row; iy++) {
        for (int ix = 0; ix < col; ix++) {
            buf(ix, iy) = rand() % 256 + std::numeric_limits<IntT>::min();
        }
    }
}
#ifdef ON_X86
void swizzle_kernel(uint8_t const *B_array, uint8_t *B_swizzled_array, int col, int acc) {
    __m256i t1, t2, t3, t4, r1, r2, r3, r4;
    for (int xoo = 0; xoo < acc; xoo += (4 * 32)) {
        for (int yo = 0; yo < col; yo += 32) {
            for (int xoi = 0; xoi < 32 * 4; xoi += 4) {
                int xo = xoo + xoi;

                t1 = _mm256_loadu_si256((__m256i_u const*)&B_array[xo * col + yo]);
                t2 = _mm256_loadu_si256((__m256i_u const*)&B_array[xo * col + yo + col]);
                t3 = _mm256_loadu_si256((__m256i_u const*)&B_array[xo * col + yo + col * 2]);
                t4 = _mm256_loadu_si256((__m256i_u const*)&B_array[xo * col + yo + col * 3]);

                r1 = _mm256_unpacklo_epi8(t1, t2);
                r2 = _mm256_unpackhi_epi8(t1, t2);
                r3 = _mm256_unpacklo_epi8(t3, t4);
                r4 = _mm256_unpackhi_epi8(t3, t4);

                t1 = _mm256_unpacklo_epi16(r1, r3);
                t2 = _mm256_unpackhi_epi16(r1, r3);
                t3 = _mm256_unpacklo_epi16(r2, r4);
                t4 = _mm256_unpackhi_epi16(r2, r4);

                r1 = _mm256_permute2x128_si256(t1, t2, 0x20);
                r2 = _mm256_permute2x128_si256(t3, t4, 0x20);
                r3 = _mm256_permute2x128_si256(t1, t2, 0x31);
                r4 = _mm256_permute2x128_si256(t3, t4, 0x31);

                _mm256_storeu_si256((__m256i_u *)&B_swizzled_array[xo * col + yo * 4], r1);
                _mm256_storeu_si256((__m256i_u *)&B_swizzled_array[xo * col + yo * 4 + 32], r2);
                _mm256_storeu_si256((__m256i_u *)&B_swizzled_array[xo * col + yo * 4 + 32 * 2], r3);
                _mm256_storeu_si256((__m256i_u *)&B_swizzled_array[xo * col + yo * 4 + 32 * 3], r4);
            }
        }
    }
}


void swizzle_kernel_unrolled(uint8_t const *B_array, uint8_t *B_swizzled_array, int col, int acc) {
    __m256i t1, t2, t3, t4, r1, r2, r3, r4;
    __m256i t5, t6, t7, t8, r5, r6, r7, r8;
    for (int xoo = 0; xoo < acc; xoo += (4 * 32)) {
        for (int yo = 0; yo < col; yo += 32) {
            for (int xoi = 0; xoi < 32 * 4; xoi += 8) {
                int xo = xoo + xoi;

                t1 = _mm256_loadu_si256((__m256i_u const *)&B_array[xo * col + yo]);
                t2 = _mm256_loadu_si256((__m256i_u const *)&B_array[xo * col + yo + col]);
                t3 = _mm256_loadu_si256((__m256i_u const *)&B_array[xo * col + yo + col * 2]);
                t4 = _mm256_loadu_si256((__m256i_u const *)&B_array[xo * col + yo + col * 3]);
                t5 = _mm256_loadu_si256((__m256i_u const *)&B_array[xo * col + yo + col * 4]);
                t6 = _mm256_loadu_si256((__m256i_u const *)&B_array[xo * col + yo + col * 5]);
                t7 = _mm256_loadu_si256((__m256i_u const *)&B_array[xo * col + yo + col * 6]);
                t8 = _mm256_loadu_si256((__m256i_u const *)&B_array[xo * col + yo + col * 7]);

                r1 = _mm256_unpacklo_epi8(t1, t2);
                r2 = _mm256_unpackhi_epi8(t1, t2);
                r3 = _mm256_unpacklo_epi8(t3, t4);
                r4 = _mm256_unpackhi_epi8(t3, t4);
                r5 = _mm256_unpacklo_epi8(t5, t6);
                r6 = _mm256_unpackhi_epi8(t5, t6);
                r7 = _mm256_unpacklo_epi8(t7, t8);
                r8 = _mm256_unpackhi_epi8(t7, t8);

                t1 = _mm256_unpacklo_epi16(r1, r3);
                t2 = _mm256_unpackhi_epi16(r1, r3);
                t3 = _mm256_unpacklo_epi16(r2, r4);
                t4 = _mm256_unpackhi_epi16(r2, r4);
                t5 = _mm256_unpacklo_epi16(r5, r7);
                t6 = _mm256_unpackhi_epi16(r5, r7);
                t7 = _mm256_unpacklo_epi16(r6, r8);
                t8 = _mm256_unpackhi_epi16(r6, r8);

                r1 = _mm256_permute2x128_si256(t1, t2, 0x20);
                r2 = _mm256_permute2x128_si256(t3, t4, 0x20);
                r3 = _mm256_permute2x128_si256(t1, t2, 0x31);
                r4 = _mm256_permute2x128_si256(t3, t4, 0x31);
                r5 = _mm256_permute2x128_si256(t5, t6, 0x20);
                r6 = _mm256_permute2x128_si256(t7, t8, 0x20);
                r7 = _mm256_permute2x128_si256(t5, t6, 0x31);
                r8 = _mm256_permute2x128_si256(t7, t8, 0x31);

                _mm256_storeu_si256((__m256i_u *)&B_swizzled_array[xo * col + yo * 4], r1);
                _mm256_storeu_si256((__m256i_u *)&B_swizzled_array[xo * col + yo * 4 + 32], r2);
                _mm256_storeu_si256((__m256i_u *)&B_swizzled_array[xo * col + yo * 4 + 32 * 2], r3);
                _mm256_storeu_si256((__m256i_u *)&B_swizzled_array[xo * col + yo * 4 + 32 * 3], r4);
                _mm256_storeu_si256((__m256i_u *)&B_swizzled_array[xo * col + col * 4 + yo * 4], r5);
                _mm256_storeu_si256((__m256i_u *)&B_swizzled_array[xo * col + col * 4 + yo * 4 + 32], r6);
                _mm256_storeu_si256((__m256i_u *)&B_swizzled_array[xo * col + col * 4 + yo * 4 + 32 * 2], r7);
                _mm256_storeu_si256((__m256i_u *)&B_swizzled_array[xo * col + col * 4 + yo * 4 + 32 * 3], r8);

            }
        }
    }
}
#endif

int main(int argc, char **argv) {
    Target target("x86-64-linux-avx512_sapphirerapids");
    // Target target("x86-64-linux");

    const int col = 16384;
    const int acc = 16384;
    // const int col = 1024;
    // const int acc = 1024;

    ImageParam B(UInt(8), 2, "B");
    Buffer<uint8_t> B_buf(col, acc);

    fill_buffer(B_buf, col, acc);
    B.set(B_buf);

    Func B_swizzled("B_swz"), B_tmp("B_swz_tmp");

    Var xi("xi"), xo("xo"), y("y");
    Var xoi("xoi"), xoo("xoo"), yi("yi"), yo("yo");
    Var vectorized("vctz");

    B_swizzled(y, xi, xo) = B(y, xo * 4 + xi);
    B_swizzled.bound(xi, 0, 4);
    B_swizzled.output_buffer().dim(0).set_stride(4);
    B_swizzled.output_buffer().dim(1).set_stride(1);
    B_swizzled.reorder_storage(xi, y);

    // B_swizzled(xi, y, xo) = B(y, xo * 4 + xi);


    B_swizzled.compute_root()
        .tile(xo, y, xoo, yo, xoi, yi, 16, 64)
        .reorder(xi, yi, xoi, yo, xoo)
        .fuse(xi, yi, vectorized)
        .vectorize(vectorized);

    B_swizzled.compile_to_lowered_stmt("tiled_matmul_with_swizzle.html", {B}, Halide::HTML, target);
    // auto time_halide = Tools::benchmark(5, 5, [&]() {
    //     B_swizzled.realize(out, target);
    // });
#ifdef ON_X86
    uint8_t *B_array = B.data();
    uint8_t *B_swizzled_array = (uint8_t *)malloc(acc * col * sizeof(uint8_t));
    memset(B_swizzled_array, 0, acc * col * sizeof(uint8_t));

    auto time_manual = Tools::benchmark(5, 5, [&]() {
        swizzle_kernel(B_array, B_swizzled_array, col, acc);
    });

    // memset(B_swizzled_array, 0, acc * col * sizeof(uint8_t));
    // auto time_manual_unrolled = Tools::benchmark(10, 10, [&]() {
    //     swizzle_kernel_unrolled(B_array, B_swizzled_array, col, acc);
    // });

    // for (int i = 0; i < 32; i++) {
    //     for (int j = 0; j < 32; j++) {
    //         std::cout << (int)B_array[j + i * col] << " ";
    //     }
    //     std::cout << "\n";
    // }
    // std::cout << "\n";
    // std::cout << "Halide:\n";
    // for (int i = 0; i < 32; i++) {
    //     for (int j = 0; j < 32; j++) {
    //         for (int k = 0; k < 4; k++) {
    //             std::cout << (int) out(k, j, i) << " ";
    //         }
    //     }
    //     std::cout << "\n";
    // }
    // std::cout << "\n";
    // std::cout << "Manual:\n";
    // for (int i = 0; i < 32; i++) {
    //     for (int j = 0; j < 32; j++) {
    //         for (int k = 0; k < 4; k++) {
    //             std::cout << (int)B_swizzled_array[k + j * 4 + i * col * 4] << " ";
    //         }
    //     }
    //     std::cout << "\n";
    // }

    // Check results
    for (int i = 0; i < acc / 4; i++) {
        for (int j = 0; j < col; j++) {
            for (int k = 0; k < 4; k++) {
                if (out(k, j, i) != B_swizzled_array[k + j * 4 + i * col * 4]) {
                    std::cout << "Error at " << i << ", " << j << ", " << k << "\n";
                    std::cout << "Val: " << (int)out(k, j, i) << " vs " << (int)B_swizzled_array[k + j * 4 + i * col * 4] << "\n";
                    return 1;
                }
            }
        }
    }

    std::cout << "Manual exec time: " << time_manual << "\n";
#endif
    // std::cout << "Manual unrolled exec time: " << time_manual_unrolled << "\n";
    // std::cout << "Halide exec time: " << time_halide << "\n";
    std::cout << "success\n";
}
