#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_test_dirs.h"

#include <iomanip>
#include <iostream>

using namespace Halide;

bool matmul_bf16(Halide::Target target) {
    (void)target;

    const int acc = 4096;

    Var x("x"), y("y");
    ImageParam A_input(Float(32), 2, "lhs");
    ImageParam B_input(Float(32), 2, "rhs");

    RDom r(0, acc, "acc");

    Func mm("matmul");
    mm(x, y) = cast<float>(0);

    if (target.has_feature(Target::AVX512_SapphireRapids)) {
        Func A("A");
        Func B("B");
        A(x, y) = cast<bfloat16_t>(A(x, y));
        B(x, y) = cast<bfloat16_t>(B(x, y));
        
        mm(x, y) += cast<float>(cast<float>(A(r.x, y))) * cast<float>(B(x, r.x));
        int tile_x = 16;
        int tile_y = 32;
        int tile_r = 16;
        Var rxi("rxi"), ryi("ryi");
        RVar rri("rri"), rro("rro");

        mm.compute_at(mm.in(), x)
            .store_in(MemoryType::AMXTile)
            .update()
            .tile(x, y, rxi, ryi, tile_x, tile_y, TailStrategy::GuardWithIf)
            .split(r.x, rro, rri, tile_r)
            .reorder({rri, rxi, ryi, rro, x, y})
            .atomic()
            .vectorize(rri)
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
    } else if (target.has_feature(Target::CUDACapability70)) {
        Func A("A");
        Func B("B");
        Var rxi("rxi"), ryi("ryi");
        RVar rri("rri"), rro("rro");

        int tile_x = 16;
        int tile_y = 16;
        int tile_r = 16;


        A(x, y) = cast<float16_t>(A_input(x, y));
        B(x, y) = cast<float16_t>(B_input(x, y));

        A.compute_root().gpu_tile(x, y, rxi, ryi, tile_x, tile_y);
        B.compute_root().gpu_tile(x, y, rxi, ryi, tile_x, tile_y);

        mm(x, y) += cast<float>(cast<float>(A(r.x, y))) * cast<float>(B(x, r.x));


        mm.compute_at(mm.in(), x)
            .store_in(MemoryType::WMMAAccumulator)
            .update()
            .split(x, x, rxi, tile_x)
            .split(y, y, ryi, tile_y)
            // .gpu_tile(x, y, rxi, ryi, tile_x, tile_y)
            .split(r.x, rro, rri, tile_r)
            .reorder({rri, rxi, ryi, rro, x, y})
            .atomic()
            .vectorize(rri)
            .vectorize(rxi)
            .vectorize(ryi)
            // .gpu_blocks(x, y)
            .gpu_threads(rro)
            ;

        Var ixi("ixi"), iyi("iyi");
        mm.compute_at(mm.in(), x)
            .tile(x, y, ixi, iyi, tile_x, tile_y)
            .vectorize(ixi)
            .vectorize(iyi);
        
        Var mmxi("mmxi"), mmyi("mmyi");
        mm.in()
            .gpu_tile(x, y, mmxi, mmyi, tile_x, tile_y)
            .vectorize(mmxi)
            .vectorize(mmyi);
    } else {
        printf("Architecture not supported");
        exit(1);
    }

    Func result = mm.in();

    result.compile_to_lowered_stmt("/tmp/matmul_flat_1x1.html", {A_input, B_input}, HTML, target);

    std::cout << "Success!\n";
    return true;
}

int main(int argc, char **argv) {
    freopen("/tmp/matmul_flat_1x1.log", "w", stderr);
    // Target target("x86-64-linux-avx512_sapphirerapids");
    // Target target("x86-64-linux-cuda_capability_70");
    Target target = get_target_from_environment().with_feature(Target::CUDA).with_feature(Target::CUDACapability70);
    // Target target = get_jit_target_from_environment();
    std::cout << target;

    printf("Running AMX (bf16)\n");
    matmul_bf16(target);
    return 0;
}
