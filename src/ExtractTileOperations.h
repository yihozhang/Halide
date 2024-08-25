#ifndef HALIDE_EXTRACT_TILE_OPERATIONS_H
#define HALIDE_EXTRACT_TILE_OPERATIONS_H

/** \file
 * Defines the lowering pass that injects calls to tile intrinsics that support
 * AMX instructions.
 */

#include "Expr.h"
#include "IRMutator.h"
#include "IRVisitor.h"

namespace Halide {
namespace Internal {

struct MemToAMX : public ExprNode<MemToAMX> {
    Expr expr;
    static Expr make(Expr expr, Type type) {
        MemToAMX *node = new MemToAMX;
        node->expr = expr;
        node->type = type;
        return node;
    }
    // This should not matter because MemToAMX is inserted only for the EqSat phase.
    static const IRNodeType _node_type = IRNodeType::Call;
};

struct AMXToMem : public ExprNode<AMXToMem> {
    Expr expr;
    static Expr make(Expr expr, Type type) {
        AMXToMem *node = new AMXToMem;
        node->expr = expr;
        node->type = type;
        return node;
    }
    // This should not matter because MemToAMX is inserted only for the EqSat phase.
    static const IRNodeType _node_type = IRNodeType::Call;
};

class EqSatIRVisitor : public IRVisitor {
public:
    using IRVisitor::visit;
    virtual void visit(const MemToAMX *e) {
        e->expr.accept(this);
    }
    virtual void visit(const AMXToMem *e) {
        e->expr.accept(this);
    }
};

class EqSatIRMutator : public IRMutator {
public:
    using IRMutator::visit;
    virtual Expr visit(const MemToAMX *e) {
        Expr value = mutate(e->expr);
        if (value.same_as(e->expr)) {
            return e;
        } else {
            return MemToAMX::make(value, e->type);
        }
    }

    virtual Expr visit(const AMXToMem *e) {
        Expr value = mutate(e->expr);
        if (value.same_as(e->expr)) {
            return e;
        } else {
            return AMXToMem::make(value, e->type);
        }
    }
};

/** Rewrite any AMX tile operations that have been stored in the AMXTile memory
 * type as intrinsic calls, to be used in the X86 backend. */
Stmt extract_tile_operations(const Stmt &s);

Stmt eqsat_extract_tile_operations(const Stmt &s);

// EqSat stuffs start here
std::string run_egglog(std::vector<std::pair<std::string, std::string>> &&);

}  // namespace Internal
}  // namespace Halide

#endif
