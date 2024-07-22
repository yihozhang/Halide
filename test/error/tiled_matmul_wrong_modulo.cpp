#include "Halide.h"
#include "halide_test_dirs.h"
#include <stdio.h>

using namespace Halide;

template<typename IntT>
void fill_buffer_a(Buffer<IntT> &buf, int row, int acc) {
    for (int iy = 0; iy < row; iy++) {
        for (int ix = 0; ix < acc; ix++) {
            buf(ix, iy) = rand() % 256 + std::numeric_limits<IntT>::min();
        }
    }
}

template<typename IntT>
void fill_buffer_b(Buffer<IntT> &buf, int col, int acc) {
    for (int iy = 0; iy < acc / 4; iy++) {
        for (int ix = 0; ix < col; ix++) {
            for (int ik = 0; ik < 4; ++ik) {
                buf(ik, ix, iy) = rand() % 256 + std::numeric_limits<IntT>::min();
            }
        }
    }
}

template<typename LhsInt8, typename RhsInt8>
bool matmul(int row, int col, int acc, int tile_x, int tile_y, int tile_r, bool validate) {
    Target target("x86-64-linux-avx512_sapphirerapids");
    Buffer<LhsInt8> A_buf(acc, row);
    Buffer<RhsInt8> B_buf(4, col, acc / 4);

    Var x("x"), y("y");
    RDom r(0, acc);

    Func mm("matmul");
    mm(x, y) = cast<int32_t>(0);
    // Mod is 3 instead of 4
    mm(x, y) += cast<int32_t>(A_buf(r, y)) * cast<int32_t>(B_buf(r % 3, x, r / 4));

    Var rxi("rxi"), ryi("ryi");
    RVar rri("rri"), rro("rro");

    mm.compute_at(mm.in(), x)
        .store_in(MemoryType::AMXTile)
        .update()
        .tile(x, y, rxi, ryi, tile_x, tile_y, TailStrategy::GuardWithIf)
        .split(r, rro, rri, tile_r)
        .reorder(rri, rxi, ryi, rro, x, y)
        .atomic()
        .vectorize(rri)
        .vectorize(rxi)
        .vectorize(ryi);

    Var ixi("ixi"), iyi("iyi");
    mm.compute_at(mm.in(), x)
        .tile(x, y, ixi, iyi, tile_x, tile_y)
        .vectorize(ixi)
        .vectorize(iyi);

    Var mmxi("mmxi"), mmyi("mmyi");
    mm.in()
        .tile(x, y, mmxi, mmyi, tile_x, tile_y)
        .vectorize(mmxi)
        .vectorize(mmyi);

    Func result = mm.in();

    if (!validate) {
        // Should err with AMX mapping failure since B buffer is not swizzled correctly
        result.compile_to_lowered_stmt("/dev/null", {A_buf, B_buf}, Halide::Text, target);
    } else {
        std::cerr << "Validating compiled program\n";

        fill_buffer_a(A_buf, row, acc);
        fill_buffer_b(B_buf, col, acc);
        Buffer<int32_t> out(col, row);
        result.realize(out);

        bool should_continue = true;
        for (int j = 0; j < row && should_continue; ++j) {
            for (int i = 0; i < col && should_continue; ++i) {
                int32_t val = 0;
                for (int k = 0; k < acc; ++k) {
                    val += static_cast<int32_t>(A_buf(k, j)) * static_cast<int32_t>(B_buf(k % 3, i, k / 4));
                }
                if (val != out(i, j)) {
                    std::cerr << "Invalid result at " << i << ", " << j << "\n"
                              << out(i, j) << " != " << val << "\n"
                              << "Matrix dims: " << row << "x" << col << "x" << acc << "\nTile dims: " << tile_x << "x" << tile_y << "x" << tile_r << "\n";
                    return false;
                }
            }
        }
    }

    return true;
}

int main(int argc, char **argv) {
    bool validate = false;
    if (argc == 2 && argv[1] == std::string("--validate")) {
        validate = true;
    }
    if (validate && !get_jit_target_from_environment().has_feature(Target::AVX512_SapphireRapids)) {
        std::cerr << "Skipping test since target does not support AMX\n";
        return 0;
    }
    matmul<int8_t, int8_t>(32, 32, 32, 8, 8, 8, validate);
}