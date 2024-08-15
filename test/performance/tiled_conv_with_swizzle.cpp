#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_test_dirs.h"
#include <cstdio>

#include <iomanip>
#include <iostream>
using namespace Halide;

int main() {
    freopen("log-tiled_conv_with_swizzle.txt", "w", stderr);
    Target target("x86-64-linux-avx512_sapphirerapids");
    // const int N = 5, CI = 128, CO = 128, W = 100, H = 80;
    const int N = 32, CI = 128, CO = 128, W = 512, H = 512;
    Var x("x"), y("y"), z("z"), c("c"), n("n");
    Var xi("xi"), yi("yi"), xo("xo"), yo("yo");
    Var ni("ni"), no("no"), ci("ci"), co("co");

    ImageParam input(BFloat(16), 4, "input");
    ImageParam filter(BFloat(16), 4, "filter");
    ImageParam bias(BFloat(16), 1, "bias");

    Func filter_swizzled("filter_swz");
    Func conv("conv");

    // Stage 1: swizzle the filter
    filter_swizzled(xi, c, y, z, xo) = filter(c, y, z, xo * 2 + xi);
    filter_swizzled.bound(xi, 0, 2);
    filter_swizzled.compute_root();

    // Stage 2: convolution
    RDom r(0, CI, 0, 3, 0, 3, "r");

    RVar rxi("rxi"), rxo("rxo");

    // conv(c, x, y, n) = bias(c);
    // conv(c, x, y, n) += filter(c, r.y, r.z, r.x) * input(r.x, x + r.y, y + r.z, n);
    // conv(c, x, y, n) = cast<float>(bias(c));
    conv(c, x, y, n) = cast<float>(0);
    conv(c, x, y, n) +=
        cast<float>(input(r.x, x + r.y, y + r.z, n)) *
            cast<float>(filter_swizzled(r.x % 2, c, r.y, r.z, r.x / 2));
    conv.reorder_storage({c, n, x, y});
    /// conv(n * x_total * y_total * n_total .. + .. ) => conv(y * x_total * n_total + ... n * c_total + c

    conv.output_buffer().dim(0).set_bounds(0, CO).set_stride(1);
    conv.output_buffer().dim(1).set_bounds(0, W).set_stride(CO);
    conv.output_buffer().dim(2).set_bounds(0, H).set_stride(CO * W);
    conv.output_buffer().dim(3).set_bounds(0, N).set_stride(CO * H * W);

    input.dim(0).set_bounds(0, CI).set_stride(1);
    input.dim(1).set_bounds(0, W + 2).set_stride(CI);
    input.dim(2).set_bounds(0, H + 2).set_stride(CI * (W + 2));
    input.dim(3).set_bounds(0, N).set_stride(CI * (W + 2) * (H + 2));

    filter.dim(0).set_bounds(0, CO).set_stride(1);
    filter.dim(1).set_bounds(0, 3).set_stride(CO);
    filter.dim(2).set_bounds(0, 3).set_stride(CO * 3);
    filter.dim(3).set_bounds(0, CI).set_stride(CO * 3 * 3);

    bias.dim(0).set_bounds(0, CO).set_stride(1);

    conv.compute_at(conv.in(), co)
        .store_in(MemoryType::AMXTile)
        // Schedule the initialization
        .tile(n, c, no, co, ni, ci, 4, 4)
        .reorder(ci, ni, co, no, x, y)
        .vectorize(ci)
        .vectorize(ni)
        // Schedule the updates
        .update()
        .tile(n, c, no, co, ni, ci, 4, 4)
        // tile r.x is 2 works, while tile r.x is 4 does not.
        .split(r.x, rxo, rxi, 2)
        .reorder(rxi, ci, ni, rxo, r.y, r.z, co, no, x, y)
        .atomic()
        .vectorize(rxi)
        .vectorize(ci)
        .vectorize(ni);

    conv.in()
        .tile(n, c, no, co, ni, ci, 4, 4)
        .reorder(ci, ni, co, no, x, y)
        .vectorize(ci)
        .vectorize(ni);

    conv.in().compile_to_lowered_stmt("tiled_conv_with_swizzle.html", {input, filter, bias}, Halide::HTML, target);
}