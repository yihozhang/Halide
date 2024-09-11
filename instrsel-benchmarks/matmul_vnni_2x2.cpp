#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_test_dirs.h"

#include <iomanip>
#include <iostream>

using namespace Halide;

bool matmul_bf16(Halide::Target target) {
    (void)target;

    const int acc = 4096;
    const int X_ACC = 2;
    const int Y_ACC = 2;

    Var x("x"), y("y");
    ImageParam A(BFloat(16), 2, "lhs");
    ImageParam B(BFloat(16), 3, "rhs");

    B.dim(1).set_stride(2);

    RDom r(0, acc, "acc");

    Func mm("matmul");
    mm(x, y) = cast<float>(0);
    mm(x, y) += cast<float>(cast<float>(A(r.x, y))) * cast<float>(B(r.x % 2, x, r.x / 2));

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

    Func result = mm.in();

    result.compile_to_lowered_stmt("/tmp/matmul_preload_vnni.html", {A, B}, HTML, target);

    std::cout << "Success!\n";
    return true;
}

int main(int argc, char **argv) {
    freopen("/tmp/matmul_preload_vnni.log", "w", stderr);
    Target target("x86-64-linux-avx512_sapphirerapids");

    printf("Running AMX (bf16)\n");
    matmul_bf16(target);
    return 0;
}
