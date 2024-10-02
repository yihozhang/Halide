#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_test_dirs.h"

#include <iomanip>
#include <iostream>

using namespace Halide;

bool matmul_bf16(Halide::Target target) {
    (void)target;

    const int acc = 4096;

    Var x("x"), y("y"), z("z");
    ImageParam A(BFloat(16), 2, "A");
    ImageParam B(BFloat(16), 2, "B");
    ImageParam C(BFloat(16), 2, "C");

    RDom r(0, acc, 0, acc, "acc");

    Func mm("matmul");
    mm(x, y) = cast<float>(0);
    // mm(x, w) = A(x, y) * B(y, z) * C(z, w)
    // A(y, z) * B(z, x) * C(x, w)
    mm(x, y) += cast<float>(cast<bfloat16_t>(cast<float>(A(r.x, y)) * cast<float>(B(r.y, r.x)))) * cast<float>(C(r.y, x));

    int tile_x = 4;
    int tile_y = 8;
    int tile_r = 4;
    Var rxi("rxi"), ryi("ryi");
    RVar rxri("rxri"), rxro("rxro");
    RVar ryri("ryri"), ryro("ryro");

    mm.compute_at(mm.in(), x)
        .store_in(MemoryType::AMXTile)
        .update()
        .tile(x, y, rxi, ryi, tile_x, tile_y, TailStrategy::GuardWithIf)
        .split(r.x, rxro, rxri, tile_r)
        .split(r.y, ryro, ryri, tile_r)
        .reorder({rxri, ryri, rxi, ryi, rxro, ryro, x, y})
        .atomic()
        .vectorize(rxri)
        .vectorize(ryri)
        .vectorize(rxi)
        .vectorize(ryi);

    Var ixi("ixi"), iyi("iyi");
    mm.compute_at(mm.in(), x)
        .tile(x, y, ixi, iyi, tile_x, tile_y)
        .vectorize(ixi)
        .vectorize(iyi);

    // schedule the consumer
    Var mmxi("mmxi"), mmyi("mmyi");
    mm.in()
        .tile(x, y, mmxi, mmyi, tile_x, tile_y)
        .vectorize(mmxi)
        .vectorize(mmyi);

    Func result = mm.in();

    result.compile_to_lowered_stmt("/tmp/matmul_flat_1x1.html", {A, B}, HTML, target);

    std::cout << "Success!\n";
    return true;
}

int main(int argc, char **argv) {
    freopen("/tmp/matmul_flat_1x1.log", "w", stderr);
    Target target("x86-64-linux-avx512_sapphirerapids");

    printf("Running AMX (bf16)\n");
    matmul_bf16(target);
    return 0;
}
