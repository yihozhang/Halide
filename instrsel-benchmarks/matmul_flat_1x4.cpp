#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_test_dirs.h"

#include <iomanip>
#include <iostream>

using namespace Halide;

bool matmul_bf16(Halide::Target target) {
    (void)target;

    const int acc = 4096;
    const int X_ACC = 1;
    const int Y_ACC = 2;

    Var x("x"), y("y");
    ImageParam A(BFloat(16), 2, "lhs");
    ImageParam B(BFloat(16), 2, "rhs");

    RDom r(0, acc, "acc");

    Func mm("matmul");
    mm(x, y) = cast<float>(0);
    mm(x, y) += cast<float>(cast<float>(A(r.x, y))) * cast<float>(B(x, r.x));

    int tile_x = 16;
    int tile_y = 32;
    int tile_r = 16;


    Var cx("cx"), cy("cy");
    Var rxi("rxi"), ryi("ryi");
    // Var rvxi("rvxi"), rvyi("rvyi");
    RVar rri("rri"), rro("rro");

    mm.compute_at(mm.in(), x)
        .store_in(MemoryType::AMXTile)
        .update()
        .tile(x, y, cx, cy, X_ACC * tile_x, Y_ACC * tile_y, TailStrategy::GuardWithIf)
        .tile(cx, cy, rxi, ryi, tile_x, tile_y)
        .split(r.x, rro, rri, tile_r)
        .reorder({rri, rxi, ryi, cy, cx, rro, x, y})
        .unroll(cx)
        .unroll(cy)
        .atomic()
        .vectorize(rri)
        .vectorize(rxi)
        .vectorize(ryi);

    Var ixi("ixi"), iyi("iyi");
    mm.compute_at(mm.in(), x)
        .tile(x, y, cx, cy, X_ACC * tile_x, Y_ACC * tile_y)
        .tile(cx, cy, ixi, iyi, tile_x, tile_y)
        // .tile(ixi, iyi, rvxi, rvyi, tile_x, tile_y)
        .unroll(cx)
        .unroll(cy)
        .vectorize(ixi)
        .vectorize(iyi);

    // schedule the consumer
    Var mmxi("mmxi"), mmyi("mmyi");
    mm.in()
        .tile(x, y, cx, cy, X_ACC * tile_x, Y_ACC * tile_y)
        .tile(cx, cy, mmxi, mmyi, tile_x, tile_y)
        .reorder(mmxi, mmyi, cy, cx, x, y)
        .unroll(cx)
        .unroll(cy)
        .vectorize(mmxi)
        .vectorize(mmyi);

    // mm.in().output_buffer().dim(2).set_bounds(0, 2);
    // mm.in().output_buffer().dim(3).set_bounds(0, 2);

    Func result = mm.in();

    // Uncomment to check the asm
    // result.compile_to_llvm_assembly(Internal::get_test_tmp_dir() + "tiled_matmul_bf16.ll", {A, B}, target);
    // result.compile_to_assembly(Internal::get_test_tmp_dir() + "tiled_matmul.s", {A, B}, target);
    result.compile_to_lowered_stmt("/tmp/matmul_flat_1x4.html", {A, B}, HTML, target);

    std::cout << "Success!\n";
    return true;
}

int main(int argc, char **argv) {
    freopen("/tmp/matmul_flat_1x4.log", "w", stderr);
    Target target("x86-64-linux-avx512_sapphirerapids");

    printf("Running AMX (bf16)\n");
    matmul_bf16(target);
    return 0;
}
