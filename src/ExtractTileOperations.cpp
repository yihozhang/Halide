#include "ExtractTileOperations.h"

#include "EqSatIRParser.h"
#include "EqSatIRPrinter.h"
#include "IRMatch.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "Util.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <sys/fcntl.h>
#include <unistd.h>
#include <unordered_set>
#include <utility>

/** \file Support extraction of AMX instructions. */

/**
 * https://asciiflow.com/#/share/eJyVUkFugzAQ%2FMrKxwoRhdAkza23SmlySHvogQsBp7FkbGSbAoryiz6nr%2BlLugZDk6ghKvJhbXZmd2b3QEScUbIQBece4XFNFVmQQ0SqiCwegtCLSI1RMBtjZGhl8BIRAHh%2BeoFVbBSr4Pq36ZOiSOBpX5cDCEikSGhuipjzun0pmdnD4%2BqtwX9%2Ffg2cLmUcTML76WyO4VAtWJ%2Ff7kIkWMEJ6gbBae2%2F3q53OHBuFBz3TS1HodPqfvUO3%2F4wO7gQag07IXqVkCuZU4VzyApuWI5BAJkdZ0K1B2ZP2%2BwJ%2FEs%2BjhKY0EYViWFSaMAaO6kypBY1hLCtDRIvMTvsekmlsc2kiGgKMw2cxqkGIyEGjn%2FlzonoIMjPUibeQX5Q1bHGisbav%2FBh2kHW2ESzdlaZkqUltaFd9UZ25TnIrIOg%2Bb7vQykLnv661GysRSaSF1k78HkHcaSbntSReLAtTL%2FscOlaI9rxYaRzzgwUOTrZeOCokLzN0TDqRYvUqtFwB6Fvqco9S5r%2BBCiqsWmNLHabzny2Y7E4PyJHcvwBx0t%2BJw%3D%3D)
 *
 *   LHS Matrix                           RHS Matrix
 *
 *      K                            conceptually      with AMX
 *  ┌────────┐
 *  │12345678│                             N             N*4
 *M │        │                            ┌──┐        ┌────────┐
 *  └────────┘                            │1 │     K/4│1234    │
 *                                        │2 │        │5678    │
 * To properly multiply 2 matrices, the   │3 │        └────────┘
 * AMX instructions perform many 4 byte  K│4 │
 * dot products, this leads to a lot of   │5 │
 * striding over 4 byte areas.            │6 │
 * Normally the row of the LHS matrix,    │7 │
 * 123... would multiply with the column  │8 │
 * of the RHS matrix 123..., but with AMX └──┘
 * this column is split up into a matrix of columns / 4 byte and rows * 4.
 * which then results in K/4 dot products per row.
 *
 */

namespace Halide {
namespace Internal {
using std::string;
using std::vector;

namespace {
template<int Dim>
struct Tile {
    bool result;
    Expr base;
    Expr stride[Dim];
    int extent[Dim];
};

enum class AMXOpType {
    Int8,
    Bfloat16,
};

/// returns the appropriate `Halide::Type` for the given operation type
Type amx_op_type_result_type(AMXOpType op_ty) {
    switch (op_ty) {
    case AMXOpType::Int8:
        return Int(32, 256);
    case AMXOpType::Bfloat16:
        return Float(32, 256);
    default:
        internal_error << "Unexpected";
        return Type();
    }
}

int amx_op_type_size(AMXOpType op_ty) {
    switch (op_ty) {
    case AMXOpType::Int8:
        return 1;
    case AMXOpType::Bfloat16:
        return 2;
    default:
        internal_error << "Unexpected";
        return -1;
    }
}

const auto wild_i32 = Variable::make(Int(32), "*");
const auto wild_i32x = Variable::make(Int(32, 0), "*");

Tile<1> get_1d_tile_index(const Expr &e) {
    if (const auto *r1 = e.as<Ramp>()) {
        return {true, r1->base, {r1->stride}, {r1->lanes}};
    }

    return {};
}

Tile<2> get_2d_tile_index(const Expr &e) {
    // ramp(ramp(base, 1, 4), x4(stride), 4)
    vector<Expr> matches;
    if (const auto *r1 = e.as<Ramp>()) {
        if (const auto *r2 = r1->base.as<Ramp>()) {
            auto ramp_2d_pattern = Ramp::make(Ramp::make(wild_i32, wild_i32, r2->lanes), Broadcast::make(wild_i32, r2->lanes), r1->lanes);
            if (expr_match(ramp_2d_pattern, e, matches)) {
                return {true, std::move(matches[0]), {std::move(matches[2]), std::move(matches[1])}, {r1->lanes, r2->lanes}};
            }
        }
    }
    return {};
}

Tile<3> get_3d_tile_index(const Expr &e) {
    vector<Expr> matches;

    // there could be a sub node
    const Sub *sub = e.as<Sub>();
    const Add *add = nullptr;

    if (sub) {
        add = sub->a.as<Add>();
    } else {
        add = e.as<Add>();
    }

    if (!add) {
        return {};
    }

    const auto &first = add->a;
    const auto &second = add->b;

    // ramp(x[x*r](base), x[x*r](stride), x) + x[x*y](ramp(idx, 1, r))

    const auto *r1 = first.as<Ramp>();
    const auto *b2 = second.as<Broadcast>();
    if (!r1 && !b2) {
        // Try switching the order
        r1 = second.as<Ramp>();
        b2 = first.as<Broadcast>();
    }
    if (!r1 || !b2) {
        return {};
    }

    const auto *b1 = r1->base.as<Broadcast>();
    const auto *r2 = b2->value.as<Ramp>();

    if (!b1 || !r2) {
        return {};
    }

    int x_tile = r1->lanes;
    int r_tile = r2->lanes;
    int y_tile = b1->lanes / r_tile;
    if (y_tile != b2->lanes / x_tile) {
        return {};
    }

    auto pattern1 = Ramp::make(Broadcast::make(wild_i32, b1->lanes), Broadcast::make(wild_i32, b1->lanes), r1->lanes);
    if (!expr_match(pattern1, first, matches)) {
        return {};
    }
    Expr base = std::move(matches[0]);
    Expr x_stride = std::move(matches[1]);

    auto pattern2 = Broadcast::make(Ramp::make(wild_i32, wild_i32, r2->lanes), b2->lanes);
    if (!expr_match(pattern2, second, matches)) {
        return {};
    }
    base += std::move(matches[0]);
    Expr r_stride = std::move(matches[1]);

    if (sub) {
        Expr adj = sub->b;
        const Broadcast *bcast = adj.as<Broadcast>();

        if (!bcast) {
            return {};
        }

        if (bcast->lanes != b1->lanes * r1->lanes) {
            return {};
        }

        base -= bcast->value;
    }

    return {true, base, {x_stride, 0, r_stride}, {x_tile, y_tile, r_tile}};
}

/**
 * \brief Get the 3d rhs tile index configuration
 *
 * \param e index expression
 * \param element_width the width of the elements, 1 for u8/i8, 2 for bf16
 * \return Tile<3> the tile configuration found
 *
 * The pattern which is getting matched looks roughly like
 * `broadcast(ramp(0, 1, r), x*y) / broadcast(4, x*y*r) + optional(broadcast(base, x*y*r)) * broadcast(8, x*y*r) +
 *  broadcast(ramp(0, 1, r), x*y) % broadcast(4, x*y*r) +
 *  broadcast(ramp(broadcast(_, r), broadcast(4, r), y) , x)`
 */
Tile<3> get_3d_rhs_tile_index(const Expr &e, int element_width) {
    const auto *sub = e.as<Sub>();
    const Add *add_lhs = nullptr;

    // there's not always a sub pattern
    // This depends on whether we have an ImageParam or a Buffer
    if (!sub) {
        add_lhs = e.as<Add>();
    } else {
        add_lhs = sub->a.as<Add>();
    }

    if (!add_lhs) {
        return {};
    }

    // The right hand side of the add expression is used for retrieving the dimensions of the matrix.
    // obtain the x, y, r dimensions
    // this expr looks like below, the shape of `add_lhs->a` can be seen further down below
    // broadcast(ramp(0, 1, r), x*y) % broadcast(4, x*y*r) + broadcast(ramp(broadcast(base, r), broadcast(4, r), y) , x)
    const Add *dim_expr = add_lhs->b.as<Add>();

    if (!dim_expr) {
        return {};
    }

    // broadcast(ramp(broadcast(_, r), broadcast(4, r), y), x)
    const Broadcast *base_stride_bc = dim_expr->b.as<Broadcast>();

    if (!base_stride_bc) {
        return {};
    }

    int tile_x = base_stride_bc->lanes;

    // broadcast(ramp(0, 1, r), x*y) % broadcast(4, x*y*r)
    std::vector<Expr> results{};
    const Expr mod_pattern = Mod::make(wild_i32x, Broadcast::make(4 / element_width, 0));
    if (!expr_match(mod_pattern, dim_expr->a, results)) {
        return {};
    }

    // broadcast(ramp(0, 1, r), x*y)
    const Broadcast *bc_ramp = results[0].as<Broadcast>();

    if (!bc_ramp) {
        return {};
    }

    int tile_xy = bc_ramp->lanes;
    int tile_y = tile_xy / tile_x;

    // ramp(0, 1, r)
    const Ramp *r_ramp = bc_ramp->value.as<Ramp>();

    if (!r_ramp) {
        return {};
    }

    int tile_r = r_ramp->lanes;

    // get the base and stride
    // ramp(broadcast(_, r), broadcast(4, r), y)
    const Expr base_stride_ramp_pattern = Ramp::make(Broadcast::make(wild_i32, tile_r), Broadcast::make(4 / element_width, tile_r), tile_y);
    if (!expr_match(base_stride_ramp_pattern, base_stride_bc->value, results)) {
        return {};
    }

    Expr base = results[0];
    Expr stride;

    bool found_stride = false;

    // the following pattern will match the following shape
    // broadcast(ramp(0, 1, k), x*y) / broadcast(4, x*y*k) * broadcast(_, x*y*k)
    // where the stride is marked by _.

    // this stride pattern can occur if `tile_r` is the same size as `acc`
    auto stride_pattern = Broadcast::make(Ramp::make(0, 1, tile_r), tile_x * tile_y) / Broadcast::make((4 / element_width), tile_x * tile_y * tile_r) * Broadcast::make(wild_i32, tile_x * tile_y * tile_r);

    if (expr_match(stride_pattern, add_lhs->a, results)) {
        found_stride = true;
        stride = std::move(results[0]);
    }

    // This pattern is similar to the above except with an additional offset to iterate over the tiles in the k dimension
    // (broadcast(ramp(0, 1, k), m * n) / broadcast(4, m*n*k) + _) * broadcast(_, m*n*k)
    // here the first _ marks the base and the second _ the stride.
    if (!found_stride) {
        stride_pattern = (Broadcast::make(Ramp::make(0, 1, tile_r), tile_x * tile_y) / Broadcast::make((4 / element_width), tile_x * tile_y * tile_r) + wild_i32) * Broadcast::make(wild_i32, tile_x * tile_y * tile_r);
        if (expr_match(stride_pattern, add_lhs->a, results)) {
            found_stride = true;
            stride = std::move(results[1]);
            base = std::move(results[0]) * stride + base;
        }
    }

    if (!found_stride) {
        return {};
    }

    return {true, base, {stride, 0, 0}, {tile_x, tile_y, tile_r}};
}

struct BaseStride {
    bool result{false};
    Expr base{};
    Expr stride{};
};

BaseStride get_rhs_tile_index(const Expr &index, int element_width, int tile_x, int tile_y, int tile_r) {
    const auto rhs_tile2 = get_2d_tile_index(index);

    if (!rhs_tile2.result) {
        const auto rhs_tile1 = get_1d_tile_index(index);

        if (!rhs_tile1.result) {
            auto rhs_tile3 = get_3d_rhs_tile_index(index, element_width);
            if (rhs_tile3.extent[0] != tile_x || rhs_tile3.extent[1] != tile_y || rhs_tile3.extent[2] != tile_r) {
                return {};
            }

            return {true, rhs_tile3.base, rhs_tile3.stride[0] * element_width};
        } else {
            // 1D: degenerate as dot product. There are two cases:
            //   * tile_r is 4, so effectively there is only one row in the loaded tile
            //   * rhs.stride.1 == 4 && tile_y = 1, where the loaded RHS has shape (K/4)x4
            //     and is contiguous in the memory
            if (rhs_tile1.extent[0] != tile_y * tile_r) {
                return {};
            }
            if (!(rhs_tile1.stride[0].as<IntImm>() && rhs_tile1.stride[0].as<IntImm>()->value == 1)) {
                return {};
            }

            if (tile_r == 4 / element_width) {
                return {true, rhs_tile1.base, 0};
            }

            if (tile_y == 1) {
                // 4 elements in u8/i8 and 2 elements for bf16.
                return {true, rhs_tile1.base, 4 / element_width};
            }

            return {};
        }
    } else {
        // The only case where there is a ramp of ramp is when tile_y = 1 and so RHS has size (K/4)x4
        // (and rhs.stride.1 != 4, for o.w. it degenerates to 1D)
        if (tile_y != rhs_tile2.extent[0] || tile_r != rhs_tile2.extent[1]) {
            return {};
        }
        if (!(rhs_tile2.stride[1].as<IntImm>() && rhs_tile2.stride[1].as<IntImm>()->value == 1)) {
            return {};
        }

        if (tile_y != 1) {
            return {};
        }

        return {true, rhs_tile2.base, rhs_tile2.stride[0]};
    }
}

struct Matmul {
    bool result = false;
    Stmt stmt;
    int tile_x;
    int tile_y;
    int tile_r;
};

Matmul convert_to_matmul(const Store *op, const string &new_name, AMXOpType op_type) {
    // m[ramp(0, 1, S)] = VectorAdd(lhs[{XYR tile}] * xX(rhs[{YR tile}])) + m[ramp(0, 1, S)]
    const auto wild_i8x = Variable::make(Int(8, 0), "*");
    const auto wild_u8x = Variable::make(UInt(8, 0), "*");
    const auto wild_bf16x = Variable::make(BFloat(16, 0), "*");
    const auto wild_f32x = Variable::make(Float(32, 0), "*");

    vector<Expr> matches;
    if (op_type == AMXOpType::Int8) {
        const auto pattern1 = wild_i32x + wild_i32x;
        if (!expr_match(pattern1, op->value, matches)) {
            return {};
        }
    } else {  // AMXOpType::Bfloat16
        const auto pattern1 = wild_f32x + wild_f32x;
        if (!expr_match(pattern1, op->value, matches)) {
            return {};
        }
    }

    const auto *reduce = matches[0].as<VectorReduce>();
    const auto *load = matches[1].as<Load>();
    if (!reduce || reduce->op != VectorReduce::Add) {
        return {};
    }
    if (!load || load->name != op->name || !equal(load->index, op->index)) {
        return {};
    }

    if (op_type == AMXOpType::Int8) {
        auto pattern2 = cast(Int(32, 0), cast(Int(32, 0), wild_i8x) * wild_i32x);
        auto pattern2_unsigned = cast(Int(32, 0), cast(Int(32, 0), wild_u8x) * wild_i32x);

        if (!(expr_match(pattern2, reduce->value, matches) || expr_match(pattern2_unsigned, reduce->value, matches))) {
            return {};
        }
    } else {
        auto pattern2 = cast(Float(32, 0), cast(Float(32, 0), wild_bf16x) * wild_f32x);

        if (!expr_match(pattern2, reduce->value, matches)) {
            return {};
        }
    }

    const auto *lhs_load = matches[0].as<Load>();
    const auto *rhs_broadcast = matches[1].as<Broadcast>();

    const Cast *rhs_cast = nullptr;

    if (lhs_load && !rhs_broadcast) {
        // now working on a larger k dimension
        // with a K dimension of 4 (or 2) with bf16 all the elements in the right-hand matrix are
        // layed out in a way that multiplying with a column can be done in a single dot product.
        // Therefore the indexing can be reused with a broadcast,
        // with higher K dimensions this can no longer be done and the broadcast won't exist.
        // ┌──┐
        // │1 │
        // │2 │
        // │3 │   ┌────────┐
        // │4 │   │1234    │
        // │5 │   │5678    │
        // │6 │   └────────┘
        // │7 │
        // │8 │
        // └──┘
        rhs_cast = matches[1].as<Cast>();
    } else {
        rhs_cast = rhs_broadcast->value.as<Cast>();
    }

    if (!lhs_load || !rhs_cast) {
        return {};
    }

    if (rhs_cast) {
        bool is_i8_u8 = rhs_cast->value.type().element_of() == Int(8) || rhs_cast->value.type().element_of() == UInt(8);
        bool is_bf16 = rhs_cast->value.type().element_of() == BFloat(16);

        if ((op_type == AMXOpType::Int8 && !is_i8_u8) || (op_type == AMXOpType::Bfloat16 && !is_bf16)) {
            user_error << "Expected rhs type of " << (op_type == AMXOpType::Int8 ? "i8/u8" : "bf16")
                       << ", got " << rhs_cast->value.type() << " instead.\nIn Expression: " << Expr(rhs_cast);
        }
    } else {
        return {};
    }

    const auto *rhs_load = rhs_cast->value.as<Load>();
    if (!rhs_load) {
        return {};
    }

    const auto lhs_tile = get_3d_tile_index(lhs_load->index);

    if (!lhs_tile.result) {
        return {};
    }

    const int tile_x = lhs_tile.extent[0];
    const int tile_y = lhs_tile.extent[1];
    const int tile_r = lhs_tile.extent[2];
    const int factor = reduce->value.type().lanes() / reduce->type.lanes();

    Expr rhs_base;
    Expr rhs_stride;

    auto opt_base_stride = get_rhs_tile_index(rhs_load->index, amx_op_type_size(op_type), tile_x, tile_y, tile_r);

    if (!opt_base_stride.result) {
        return {};
    }

    rhs_base = opt_base_stride.base;
    rhs_stride = opt_base_stride.stride;

    if (op->index.type().lanes() != tile_x * tile_y ||
        factor != tile_r) {
        return {};
    }

    // {rows, colbytes, var, index}
    auto lhs_var = Variable::make(Handle(), lhs_load->name);
    const auto &lhs_load_type = lhs_load->type;
    int element_width = lhs_load_type.bytes();
    auto lhs_type = lhs_load_type.with_lanes(1024 / element_width);
    auto lhs = Call::make(lhs_type, "tile_load", {tile_x, tile_r * element_width, lhs_var, lhs_tile.base * element_width, lhs_tile.stride[0] * element_width}, Call::Intrinsic);

    auto rhs_var = Variable::make(Handle(), rhs_load->name);
    const auto &rhs_load_type = rhs_load->type;
    auto rhs_type = rhs_load_type.with_lanes(1024 / element_width);

    auto rhs = Call::make(rhs_type, "tile_load", {tile_r / (4 / element_width), tile_y * 4, rhs_var, rhs_base * element_width, rhs_stride}, Call::Intrinsic);
    auto res_type = amx_op_type_result_type(op_type);

    // {rows, colbytes, acc, out, lhs, rhs}
    auto out = Load::make(res_type, new_name, Ramp::make(0, 1, 256), {}, {}, const_true(256), {});

    // 4 bytes for i32, f32
    auto colbytes = tile_y * 4;
    auto matmul = Call::make(res_type, "tile_matmul", {tile_x, colbytes, tile_r, out, lhs, rhs}, Call::Intrinsic);
    auto store = Store::make(new_name, matmul, Ramp::make(0, 1, 256), Parameter(), const_true(256), ModulusRemainder());
    return {true, std::move(store), tile_x, tile_y, tile_r};
}

Stmt convert_to_zero(const Store *op, int tile_x, int tile_y, const string &new_name) {
    if (const auto *ramp = op->index.as<Ramp>()) {
        if (const auto *bcast = op->value.as<Broadcast>()) {
            if (is_const_one(ramp->stride) &&
                is_const_zero(bcast->value) &&
                (bcast->lanes == tile_x * tile_y)) {
                auto rows = Cast::make(Int(16), tile_x);
                auto bytes = op->value.type().bytes();
                auto colbytes = Cast::make(Int(16), tile_y * bytes);
                const auto &store_type = op->value.type();
                // will be f32 or i32
                auto tile_zero_type = store_type.with_lanes(1024 / store_type.bytes());
                auto val = Call::make(tile_zero_type, "tile_zero", {rows, colbytes}, Call::Intrinsic);
                auto store = Store::make(new_name, std::move(val), Ramp::make(0, 1, 256), Parameter(), const_true(256), ModulusRemainder());
                return store;
            }
        }
    }
    return {};
}

Stmt convert_to_tile_store(const Store *op, const string &amx_name, int tile_x, int tile_y) {
    auto tile = get_2d_tile_index(op->index);
    if (tile.result && tile.extent[0] == tile_x && tile.extent[1] == tile_y) {
        auto out = Variable::make(Handle(), op->name);
        auto tile_type = op->value.type().with_lanes(256);
        auto tile_val = Load::make(tile_type, amx_name, Ramp::make(0, 1, 256), {}, {}, const_true(256), {});
        auto bytes = op->value.type().bytes();
        internal_assert(bytes == 4) << "AMX store only supported for int32 and float32 output, not for " << op->value.type() << "\n";
        // {tile_x, tile_y, var, base, stride}
        auto store = Call::make(Int(32), "tile_store", {tile_x, tile_y * bytes, std::move(out), tile.base * bytes, tile.stride[0] * bytes, std::move(tile_val)}, Call::Intrinsic);
        return Evaluate::make(std::move(store));
    }
    return {};
}

class ExtractTileOperations : public IRMutator {
    using IRMutator::visit;

    string tile_name;
    string amx_name;
    vector<Stmt> pending_stores;
    bool in_allocate = false;
    int found_tile_x = -1;
    int found_tile_y = -1;
    int found_tile_r = -1;
    AMXOpType op_type;

    Stmt visit(const Allocate *op) override {
        if (op->memory_type == MemoryType::AMXTile) {
            user_assert(
                (op->type.is_int() && op->type.bits() == 32) ||
                (op->type.is_float() && op->type.bits() == 32))
                << "scheduled tile operations must yield 32-bit integers or 32-bit floats";

            if (op->type.is_int() && op->type.bits() == 32) {
                op_type = AMXOpType::Int8;
            } else {
                op_type = AMXOpType::Bfloat16;
            }

            user_assert(!in_allocate) << "Already in AMX allocation: " << amx_name;
            ScopedValue<string> old_amx_name(amx_name, op->name + ".amx");
            ScopedValue<string> old_tile_name(tile_name, op->name);
            ScopedValue<bool> old_in_alloc(in_allocate, true);
            Stmt body = op->body;

            pending_stores.clear();
            body = mutate(body);
            if (found_tile_x < 0 || found_tile_y < 0 || found_tile_r < 0) {
                return op;
            }
            if (!pending_stores.empty()) {
                // Really only need to go over the pending stores
                body = mutate(body);
            }

            auto tile_type = amx_op_type_result_type(op_type);
            return Allocate::make(amx_name, tile_type, MemoryType::AMXTile, {1}, const_true(), body);
        }
        return IRMutator::visit(op);
    }

    Stmt visit(const Free *op) override {
        if (op->name != tile_name) {
            return op;
        }
        return Free::make(amx_name);
    }

    Stmt visit(const ProducerConsumer *op) override {
        if (op->name != tile_name) {
            return IRMutator::visit(op);
        }

        auto body = mutate(op->body);
        return ProducerConsumer::make(amx_name, op->is_producer, std::move(body));
    }

    Expr visit(const Load *op) override {
        // Any tile load will be matched elsewhere, so a load here means that
        // the AMX tile is used outside of a tile instruction.
        user_assert(op->name != tile_name) << "AMX tile allocation used outside a tile instruction";
        return IRMutator::visit(op);
    }

    Stmt visit(const Store *op) override {
        if (op->name != tile_name) {
            const auto *load = op->value.as<Load>();
            if (!load || load->name != tile_name) {
                return op;
            }
            auto store = convert_to_tile_store(op, amx_name, found_tile_x, found_tile_y);
            user_assert(store.defined()) << "Store to AMX tile allocation of a non-tile value";
            return store;
        }

        auto matmul = convert_to_matmul(op, amx_name, op_type);
        if (matmul.result) {
            user_assert(
                (found_tile_x < 0 || matmul.tile_x == found_tile_x) &&
                (found_tile_y < 0 || matmul.tile_y == found_tile_y) &&
                (found_tile_r < 0 || matmul.tile_r == found_tile_r))
                << "Found different tile sizes for AMX tile allocation";
            found_tile_x = matmul.tile_x;
            found_tile_y = matmul.tile_y;
            found_tile_r = matmul.tile_r;

            return matmul.stmt;
        }

        if (found_tile_x < 0 || found_tile_y < 0) {
            pending_stores.emplace_back(op);
            return op;
        }

        auto zero = convert_to_zero(op, found_tile_x, found_tile_y, amx_name);
        if (zero.defined()) {
            return zero;
        }

        // Otherwise there is some other operation using the allocation, so we cannot use the AMX instructions
        user_error << "Found non-tile operations for AMX tile allocation";
        return op;
    }
};

}  // namespace

namespace EqSatExtensions {

Location from_memory_type(MemoryType memtype) {
    if (memtype == MemoryType::AMXTile) {
        return Location::AMX;
    } else if (memtype == MemoryType::WMMAAccumulator) {
        return Location::WMMA;
    } else {
        return Location::Mem;
    }
}

Expr GLoad::make(Type type, std::shared_ptr<Var> name, Expr index, Buffer<> image, Parameter param, Expr predicate, ModulusRemainder alignment) {
    internal_assert(predicate.defined()) << "Load with undefined predicate\n";
    internal_assert(index.defined()) << "Load of undefined\n";
    internal_assert(type.lanes() == index.type().lanes()) << "Vector lanes of Load must match vector lanes of index\n";
    internal_assert(type.lanes() == predicate.type().lanes())
        << "Vector lanes of Load must match vector lanes of predicate\n";

    GLoad *node = new GLoad();
    node->type = type;
    node->name = std::move(name);
    node->predicate = std::move(predicate);
    node->index = std::move(index);
    node->image = std::move(image);
    node->param = std::move(param);
    node->alignment = alignment;
    return node;
}

Expr GVariable::make(Type type, std::shared_ptr<Var> name) {
    GVariable *node = new GVariable;
    node->type = type;
    node->name = std::move(name);
    return node;
}

// Inserting LocToLoc nodes to represent the data movement between different memory types
// The only case we need to insert an AMXToMem node is when we are loading to an AMX tile
// The only case we need to insert a MemToAMX node is when we are storing to an AMX tile
class AnnotateDataMovement : public IRMutator {
    using IRMutator::visit;

public:
    AnnotateDataMovement() = default;

protected:
    std::map<string, MemoryType> vars_memory_type;

    Stmt visit(const Allocate *op) override {
        if (op->memory_type == MemoryType::AMXTile) {
            user_assert(
                (op->type.is_int_or_uint() && op->type.bits() == 32) ||
                (op->type.is_float() && op->type.bits() == 32) ||
                (op->type.is_int_or_uint() && op->type.bits() == 8) ||
                (op->type.is_bfloat() && op->type.bits() == 16))
                << "scheduled tile operations must yield 32-bit integers or 32-bit floats for output, 8-bit integers or 16-bit floats for input";
            user_assert(vars_memory_type.find(op->name) == vars_memory_type.end()) << "Duplicate variable name: " << op->name;

        } else if (op->memory_type == MemoryType::WMMAAccumulator) {
            user_assert(op->type.is_float() && op->type.bits() == 32) << "currently only support float16xfloat16 to float32 for WMMA";

        } else {
            return IRMutator::visit(op);
        }

        std::map<string, MemoryType> curr_vars_memory_type(this->vars_memory_type);
        curr_vars_memory_type[op->name] = op->memory_type;
        ScopedValue<std::map<string, MemoryType>> old_tile_vars(vars_memory_type, std::move(curr_vars_memory_type));

        Stmt body = mutate(op->body);

        return Allocate::make(op->name, op->type, op->memory_type, op->extents, const_true(), body);
    }

    Expr visit(const Load *load) override {
        auto result = vars_memory_type.find(load->name);
        if (result != vars_memory_type.end()) {
            internal_assert(is_const_one(load->predicate)) << "Only constant predicate is supported in accelerator tiles";
            return LocToLoc::make(load, load->type, from_memory_type(result->second), Location::Mem);
        } else {
            return load;
        }
    }

    Stmt visit(const Store *store) override {
        auto result = vars_memory_type.find(store->name);
        if (result != vars_memory_type.end()) {
            // `Load` can only occur in store->value,
            Expr value = LocToLoc::make(mutate(store->value), store->value.type(), Location::Mem, from_memory_type(result->second));
            internal_assert(is_const_one(store->predicate)) << "Only constant predicate is supported";
            return Store::make(
                store->name,
                value,
                store->index,
                store->param,
                store->predicate,
                store->alignment);
        } else {
            return IRMutator::visit(store);
        }
    }
};

std::string PLACEHOLDER_PREFIX = "collectstoresplaceholder";

struct CollectStores : public EqSatIRMutator {
    using EqSatIRMutator::visit;
    std::map<std::string, Stmt> stores;

    Stmt visit(const Store *op) override {
        auto no = stores.size();
        auto placeholder = PLACEHOLDER_PREFIX + op->name + ".eqsat." + std::to_string(no);
        placeholder = replace_all(placeholder, "$", "AAA");
        placeholder = replace_all(placeholder, ".", "BBB");
        stores[placeholder] = op;
        return Store::make(placeholder, op->value, op->index, op->param, op->predicate, op->alignment);
    }
};

class RemoveGLoadsAndGVars : public EqSatIRMutator {
    using EqSatIRMutator::visit;

public:
    struct PrologueStmt {
        std::string name;
        Expr expr;
    };
    RemoveGLoadsAndGVars(const std::string &prefix)
        : prefix(prefix) {
    }

    const std::vector<PrologueStmt> &get_prologues() {
        return this->prologues;
    }

protected:
    std::string prefix;
    int store_no = 0;
    std::vector<PrologueStmt> prologues;
    Expr visit(const GLoad *load) override {
        if (const StringVar *v = load->name.get()->to_string_var()) {
            return Load::make(load->type, v->name, EqSatIRMutator::mutate(load->index), load->image, load->param, EqSatIRMutator::mutate(load->predicate), load->alignment);
        } else if (const ExprVar *v = load->name.get()->to_expr_var()) {
            // We traverse children first to ensure that the prologues are in the correct order
            Expr predicate = EqSatIRMutator::mutate(load->predicate);
            Expr index = EqSatIRMutator::mutate(load->index);

            std::string name = prefix + ".temporary." + std::to_string(store_no++);
            prologues.push_back({name, mutate(v->expr)});
            return Load::make(load->type, name, index, load->image, load->param, predicate, load->alignment);
        } else {
            internal_error << "GLoad name must be a StringVar or ExprVar\n";
            return {};
        }
    }

    Expr visit(const GVariable *var) override {
        if (const StringVar *v = var->name.get()->to_string_var()) {
            return Variable::make(var->type, v->name);
        } else if (const ExprVar *v = var->name.get()->to_expr_var()) {
            auto name = prefix + ".temporary." + std::to_string(store_no++);
            prologues.push_back({name, mutate(v->expr)});
            return Variable::make(var->type, name);
        } else {
            internal_error << "GVariable name must be a StringVar\n";
            return {};
        }
    }
};

struct SubstStores : public EqSatIRMutator {
    using EqSatIRMutator::visit;
    const std::map<std::string, Stmt> &stores;

    SubstStores(std::map<std::string, Stmt> &&stores)
        : stores(stores) {
    }

    Stmt visit(const Store *op) override {
        internal_assert(op->name.find(PLACEHOLDER_PREFIX) == 0)
            << "All stores should have been replaced with a place holder";
        auto it = stores.find(op->name);
        internal_assert(it != stores.end()) << "Store not found";

        Stmt s = it->second;
        auto prefix = replace_all(op->name.substr(PLACEHOLDER_PREFIX.length()), "AAA", "$");
        prefix = replace_all(prefix, "BBB", ".");
        RemoveGLoadsAndGVars remover(prefix);
        s = remover.mutate(s);

        vector<Stmt> stores;
        for (const auto &prologue : remover.get_prologues()) {
            int lanes = prologue.expr.type().lanes();
            stores.push_back(Store::make(prologue.name, prologue.expr, Ramp::make(0, 1, lanes), Parameter(), const_true(lanes), ModulusRemainder()));
        }
        stores.push_back(s);
        s = Block::make(stores);
        for (auto i = remover.get_prologues().rbegin(); i != remover.get_prologues().rend(); i++) {
            s = Allocate::make(i->name, i->expr.type(), MemoryType::Auto, {1}, const_true(i->expr.type().lanes()), s);
        }

        return s;
    }
};

Type convert_to_tile_type(Type type) {
    Type tile_type;
    if (type.is_bfloat() && type.bits() == 16) {
        tile_type = BFloat(16, 512);
    } else if (type.is_int() || type.is_uint() || type.is_float()) {  // is_float() returns true for bfloat
        tile_type = type.with_lanes(1024 / type.bytes());
    } else {
        internal_error << "Unsupported type for AMX tile";
    }
    return tile_type;
}

// The expression returned by EqSat, is not necessarily compatiable with the
// runtime, since the runtime always views an AMX Tile as a 256-lane vector.
class EnforceAMXShape : public IRMutator {
    using IRMutator::visit;

protected:
    std::vector<string> tile_vars;
    std::unordered_map<string, int> amx_tile_size;

    Stmt visit(const Allocate *op) override {
        if (op->memory_type == MemoryType::AMXTile) {
            std::vector<string> curr_tile_vars(this->tile_vars);
            curr_tile_vars.push_back(op->name);
            ScopedValue<vector<string>> old_tile_vars(tile_vars, std::move(curr_tile_vars));
            auto body = mutate(op->body);
            internal_assert(amx_tile_size.count(op->name)) << "AMX tile size not found";

            auto tile_type = convert_to_tile_type(op->type);
            return Allocate::make(op->name, tile_type, MemoryType::AMXTile, {amx_tile_size[op->name]}, const_true(), body);
        } else {
            return IRMutator::visit(op);
        }
    }

    int get_nth_tile_from_tile_index(const Expr &e) {
        const Ramp *ramp = e.as<Ramp>();
        if (!ramp) {
            internal_error << "AMX tile can only be indexed with ramp";
            return -1;
        }
        int vec_length = ramp->lanes;
        const int64_t *basep = as_const_int(ramp->base);
        if (!basep) {
            internal_error << "Only constant base is supported in AMX";
            return -1;
        }
        int base = (int)*basep;
        internal_assert(base % vec_length == 0) << "Cannot determine which AMX tile to load from";
        return base / vec_length;
    }

    Expr visit(const Load *load) override {
        if (std::find(tile_vars.begin(), tile_vars.end(), load->name) != tile_vars.end()) {
            int tile = get_nth_tile_from_tile_index(load->index);
            amx_tile_size[load->name] = std::max(amx_tile_size[load->name], tile + 1);
            Type tile_type = convert_to_tile_type(load->type);
            int tile_lanes = tile_type.lanes();
            return Load::make(tile_type,
                              load->name,
                              Ramp::make(tile_lanes * tile, 1, tile_lanes),
                              load->image,
                              load->param,
                              const_true(tile_lanes),
                              load->alignment);
        } else {
            return load;
        }
    }

    Stmt visit(const Store *store) override {
        if (std::find(tile_vars.begin(), tile_vars.end(), store->name) != tile_vars.end()) {
            internal_assert(is_const_one(store->predicate)) << "Only constant predicate is supported";
            int tile = get_nth_tile_from_tile_index(store->index);
            amx_tile_size[store->name] = std::max(amx_tile_size[store->name], tile + 1);
            int tile_lanes = convert_to_tile_type(store->value.type()).lanes();

            // There should not be a Load in places other than value,
            // so we don't need to mutate them.
            Expr value = mutate(store->value);
            return Store::make(
                store->name,
                value,
                Ramp::make(tile * tile_lanes, 1, tile_lanes),
                store->param,
                const_true(tile_lanes),
                store->alignment);
        } else {
            return IRMutator::visit(store);
        }
    }

    Expr visit(const Call *call) override {
        // Not included:
        // - tile_store since it returns a single value
        vector<string> intrinsic_names = {"tile_load", "tile_matmul", "tile_zero"};

        if (std::find(intrinsic_names.begin(), intrinsic_names.end(), call->name) != intrinsic_names.end()) {
            return Call::make(
                convert_to_tile_type(call->type),
                call->name,
                mutate(call->args),
                call->call_type,
                call->func,
                call->value_index,
                call->image,
                call->param);
        } else {
            return IRMutator::visit(call);
        }
    }
};

// The intrinsics for WMMA operations are internally represented as a single-thread operation.
// However, it should be a warp-level instruction. This pass wraps the WMMA operation with GPU_lanes
// and updates the types.
// More concretely, this pass
// (1) removes the gpu_thread(idx, 0, 1) loop and replaces it with a gpu_lane(idx, 0, 32) loop
// (2) shrinks the type of the WMMA intrinsics to be per lane
// This pass shares many code with EnforceAMXShape, but for now we keep them separate.
class EnforceWMMALanes : public IRMutator {
    using IRMutator::visit;

    std::vector<string> tile_vars;
    vector<string> intrinsic_names = {
        "wmma.load.a.sync.aligned.row.m16n16k16.f16",
        "wmma.load.b.sync.aligned.row.m16n16k16.f16",
        "wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32",
        "wmma.load.c.sync.aligned.row.m16n16k16.f32",
        // "wmma.store.d.sync.aligned.row.m16n16k16.f32"
    };

protected:
    bool in_wmma = false;

    Stmt visit(const Allocate *op) override {
        if (op->memory_type == MemoryType::WMMAAccumulator) {;
            internal_assert(op->type.lanes() == 1);
            internal_assert(op->extents.size() == 1);
            internal_assert(op->extents[0].as<IntImm>()->value % 32 == 0);
            ScopedValue<bool> old_in_wmma(in_wmma, true);

            std::vector<string> curr_tile_vars(this->tile_vars);
            curr_tile_vars.push_back(op->name);
            ScopedValue<vector<string>> old_tile_vars(tile_vars, std::move(curr_tile_vars));

            auto body = mutate(op->body);
            auto lanes = op->extents[0].as<IntImm>()->value / 32;
            return Allocate::make(op->name,
                                  op->type.with_lanes(lanes),
                                  MemoryType::WMMAAccumulator, op->extents, const_true(),
                                  For::make("wmma_warp_idx", 0, 32, ForType::GPULane, Partition::Never, DeviceAPI::CUDA,
                                            body));
        } else {
            return IRMutator::visit(op);
        }
    }

    Stmt visit(const For *op) override {
        if (in_wmma && op->for_type == ForType::GPUThread) {
            auto extent = op->extent.as<IntImm>();
            auto min = op->min.as<IntImm>();
            internal_assert(extent && min && extent->value == 1 && min->value == 0) << "Intrinsics for WMMA operation should be presented as a single-thread operation";

            return mutate(op->body);
        }

        return IRMutator::visit(op);
    }

    Expr visit(const Call *op) override {
        if (std::find(intrinsic_names.begin(), intrinsic_names.end(), op->name) != intrinsic_names.end()) {
            internal_assert(in_wmma);
            internal_assert(op->type.lanes() % 32 == 0);
            return Call::make(
                op->type.with_lanes(op->type.lanes() / 32),
                op->name,
                mutate(op->args),
                op->call_type,
                op->func,
                op->value_index,
                op->image,
                op->param);
        } else {
            return IRMutator::visit(op);
        }
    }

    Expr visit(const Load *load) override {
        if (std::find(tile_vars.begin(), tile_vars.end(), load->name) != tile_vars.end()) {
            internal_assert(in_wmma);
            internal_assert(load->type.lanes() % 32 == 0);
            int tile_lanes = load->type.lanes() / 32;
            Type tile_type = load->type.with_lanes(tile_lanes);
            return Load::make(tile_type,
                              load->name,
                              Ramp::make(0, 1, tile_lanes),
                              load->image,
                              load->param,
                              const_true(tile_lanes),
                              load->alignment);
        } else {
            return load;
        }
    }

    Stmt visit(const Store *store) override {
        if (std::find(tile_vars.begin(), tile_vars.end(), store->name) != tile_vars.end()) {
            std::cerr << store->name << " with type " << store->value.type() << "\n";
            internal_assert(in_wmma);
            internal_assert(is_const_one(store->predicate)) << "Only constant predicate is supported";
            internal_assert(store->value.type().lanes() % 32 == 0);
            int tile_lanes = store->value.type().lanes() / 32;

            // There should not be a Load in places other than value,
            // so we don't need to mutate them.
            Expr value = mutate(store->value);
            return Store::make(
                store->name,
                value,
                Ramp::make(0, 1, tile_lanes),
                store->param,
                const_true(tile_lanes),
                store->alignment);
        } else {
            return IRMutator::visit(store);
        }
    }
};

class DesugarIntrinsics : public IRMutator {
    using IRMutator::visit;

protected:
    Expr visit(const Call *call) override {
        if (call->name == "KWayInterleave") {
            std::vector<Expr> args = call->args;
            internal_assert(args.size() == 3);
            const int64_t *pk = as_const_int(args[0]);
            Expr vec = mutate(args[1]);
            const int64_t *planes = as_const_int(args[2]);
            if (!pk || !planes) {
                internal_error << "KWayInterleave requires constant k and lanes\n";
                return {};
            }
            int K = *pk;
            int lanes = *planes;
            internal_assert(vec.type().lanes() % lanes == 0) << "Vector size must be a multiple of #lanes\n";
            internal_assert(lanes % K == 0) << "Number of lanes must be a multiple of k\n";
            int length = vec.type().lanes() / lanes;

            std::vector<int> indices;
            for (int i = 0; i < lanes; i += K) {
                for (int j = 0; j < length; j++) {
                    for (int k = 0; k < K; k++) {
                        indices.push_back((i + k) * length + j);
                    }
                }
            }
            return Shuffle::make({vec}, indices);

        } else {
            return IRMutator::visit(call);
        }
    }
};

}  // namespace EqSatExtensions

Stmt extract_tile_operations(const Stmt &s) {
    return ExtractTileOperations().mutate(s);
}

Stmt eqsat_extract_tile_operations(const Stmt &s) {
    auto annotated_s = EqSatExtensions::AnnotateDataMovement().mutate(s);
    EqSatExtensions::CollectStores collect_stores;
    auto result = collect_stores.mutate(annotated_s);
    auto stores = std::move(collect_stores.stores);

    std::vector<std::pair<std::string, std::string>> bindings;

    for (const auto &[name, op] : stores) {
        std::ostringstream oss;
        EqSatIRPrinter sprinter(oss);
        sprinter.print(op);
        bindings.emplace_back(name, oss.str());
    }

    auto output = run_egglog(std::move(bindings));
    auto optimized_programs = split_string(output, "\n");
    bool amx_synthesized = false;
    std::map<std::string, Stmt> new_stores;
    for (size_t i = 0; i < optimized_programs.size(); i++) {
        if (optimized_programs[i].empty()) {
            continue;
        }
        auto &optimized = optimized_programs[i];
        auto &name = bindings[i].first;
        amx_synthesized = amx_synthesized || optimized.find("tile_matmul") != string::npos;

        EqSatExtensions::EqSatIRParser parser(optimized);
        auto opvalue = parser.parse_stmt();

        new_stores[name] = opvalue;
    }
    if (!amx_synthesized) {
        std::cerr << "amx NOT synthesized\n";
    } else {
        std::cerr << "amx synthesized\n";
    }
    result = EqSatExtensions::SubstStores(std::move(new_stores)).mutate(result);
    result = EqSatExtensions::EnforceAMXShape().mutate(result);
    result = EqSatExtensions::EnforceWMMALanes().mutate(result);
    result = EqSatExtensions::DesugarIntrinsics().mutate(result);
    return result;
}

std::string run_egglog(std::vector<std::pair<std::string, std::string>> &&binding) {
#include "egglog/main.tmpl.h"

    std::string egglog_prog = EGGLOG_PROG(std::move(binding));

    std::string filename = "/tmp/egglog_prog_" + std::to_string(getpid()) + ".egg";
    // Write the program to a file
    std::cout << "Writing egglog program to " << filename << std::endl;
    std::ofstream file(filename);
    file << egglog_prog << std::flush;

    int pipe_stdin[2];
    int pipe_stdout[2];

    if (pipe(pipe_stdin) < 0 || pipe(pipe_stdout) < 0) {
        internal_error << "Failed to create pipe for egglog";
        return "";
    }

    pid_t pid = fork();

    if (pid < 0) {
        internal_error << "Failed to exec egglog";
    }

    if (pid == 0) {
        const char *argv[] = {"egglog-halide-sidecar", nullptr};
        close(pipe_stdin[1]);
        close(pipe_stdout[0]);

        // Redirect stdin and stdout
        dup2(pipe_stdin[0], STDIN_FILENO);
        dup2(pipe_stdout[1], STDOUT_FILENO);
        int devnull = open("/dev/null", O_WRONLY | O_CREAT, 0666);
        dup2(devnull, STDERR_FILENO);
        close(pipe_stdin[0]);
        close(pipe_stdout[1]);
        execvp(argv[0], const_cast<char **>(argv));
        internal_error << "egglog failed to exec";
        return "";
    }

    close(pipe_stdout[1]);

    // Write to the subprocess's stdin
    write(pipe_stdin[1], egglog_prog.c_str(), egglog_prog.size());
    close(pipe_stdin[1]);

    // Read from the subprocess's stdout
    std::ostringstream oss;
    char buffer[128];
    ssize_t count;
    while ((count = read(pipe_stdout[0], buffer, sizeof(buffer) - 1)) > 0) {
        buffer[count] = '\0';  // Null-terminate the string
        oss << buffer;
    }

    close(pipe_stdout[0]);
    return oss.str();
}

template<>
void ExprNode<EqSatExtensions::LocToLoc>::accept(IRVisitor *v) const {
    using namespace EqSatExtensions;
    EqSatIRVisitor *ev = (EqSatIRVisitor *)v;
    internal_assert(!v->is_base_ir_visitor()) << "LocToLoc can only be visited by EqSatIRVisitor\n";
    ev->visit((const LocToLoc *)this);
}
template<>
void ExprNode<EqSatExtensions::GLoad>::accept(IRVisitor *v) const {
    using namespace EqSatExtensions;
    EqSatIRVisitor *ev = (EqSatIRVisitor *)v;
    internal_assert(!v->is_base_ir_visitor()) << "GLoad can only be mutated by EqSatIRVisitor\n";
    ev->visit((const GLoad *)this);
}
template<>
void ExprNode<EqSatExtensions::Computed>::accept(IRVisitor *v) const {
    using namespace EqSatExtensions;
    EqSatIRVisitor *ev = (EqSatIRVisitor *)v;
    internal_assert(!v->is_base_ir_visitor()) << "Computed can only be mutated by EqSatIRVisitor\n";
    ev->visit((const Computed *)this);
}

template<>
void ExprNode<EqSatExtensions::GVariable>::accept(IRVisitor *v) const {
    using namespace EqSatExtensions;
    EqSatIRVisitor *ev = (EqSatIRVisitor *)v;
    internal_assert(!v->is_base_ir_visitor()) << "GVariable can only be mutated by EqSatIRVisitor\n";
    ev->visit((const GVariable *)this);
}

template<>
Expr ExprNode<EqSatExtensions::LocToLoc>::mutate_expr(IRMutator *v) const {
    using namespace EqSatExtensions;
    EqSatIRMutator *ev = (EqSatIRMutator *)v;
    internal_assert(!v->is_base_ir_mutator()) << "LocToLoc can only be mutated by EqSatIRMutator\n";
    return ev->visit((const LocToLoc *)this);
}

template<>
Expr ExprNode<EqSatExtensions::GLoad>::mutate_expr(IRMutator *v) const {
    using namespace EqSatExtensions;
    EqSatIRMutator *ev = (EqSatIRMutator *)v;
    internal_assert(!v->is_base_ir_mutator()) << "GLoad can only be mutated by EqSatIRMutator\n";
    return ev->visit((const GLoad *)this);
}

template<>
Expr ExprNode<EqSatExtensions::GVariable>::mutate_expr(IRMutator *v) const {
    using namespace EqSatExtensions;
    EqSatIRMutator *ev = (EqSatIRMutator *)v;
    internal_assert(!v->is_base_ir_mutator()) << "GVariable can only be mutated by EqSatIRMutator\n";
    return ev->visit((const GVariable *)this);
}

template<>
Expr ExprNode<EqSatExtensions::Computed>::mutate_expr(IRMutator *v) const {
    using namespace EqSatExtensions;
    EqSatIRMutator *ev = (EqSatIRMutator *)v;
    internal_assert(!v->is_base_ir_mutator()) << "Computed can only be mutated by EqSatIRMutator\n";
    return ev->visit((const Computed *)this);
}

}  // namespace Internal
}  // namespace Halide
