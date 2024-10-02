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

namespace EqSatExtensions {
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
        node->expr = std::move(expr);
        node->type = type;
        return node;
    }
    // This should not matter because MemToAMX is inserted only for the EqSat phase.
    static const IRNodeType _node_type = IRNodeType::Call;
};

// Computed is a placeholder for computed extent of ExprVars
struct Computed : public ExprNode<Computed> {
    static Expr make(Type type) {
        Computed *node = new Computed;
        node->type = Int(64);
        return node;
    }
    static const IRNodeType _node_type = IRNodeType::Call;
};

struct StringVar;
struct ExprVar;

struct Var {
    virtual ~Var() = default;
    virtual StringVar *to_string_var() { return nullptr; }
    virtual ExprVar *to_expr_var() { return nullptr; }
};

struct StringVar : public Var {
    std::string name;
    StringVar(const std::string &name)
        : name(name) {
    }
    StringVar *to_string_var() override {
        return (StringVar *) this;
    }
};

struct ExprVar : public Var {
    enum Location {
        Mem,
        AMX
    } loc;
    Expr expr;
    ExprVar(Location loc, Expr expr)
        : loc(loc), expr(std::move(expr)) {
    }

    ExprVar *to_expr_var() override {
        return (ExprVar *) this;
    }
};

// Generalized load where the buffer can be either a var (string)
// or an expression.
struct GLoad : public ExprNode<GLoad> {

    std::shared_ptr<Var> name;

    Expr predicate, index;

    Buffer<> image;
    Parameter param;
    ModulusRemainder alignment;

    static Expr make(Type type, std::shared_ptr<Var> name,
                     Expr index, Buffer<> image,
                     Parameter param,
                     Expr predicate,
                     ModulusRemainder alignment);

    static const IRNodeType _node_type = IRNodeType::Load;
};

struct GVariable : public ExprNode<GVariable> {
    std::shared_ptr<Var> name;
    static Expr make(Type type, std::shared_ptr<Var> name);
    static const IRNodeType _node_type = IRNodeType::Variable;
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
    virtual void visit(const GLoad *e) {
        e->predicate.accept(this);
        e->index.accept(this);
    }
    virtual void visit(const GVariable *e) {
    }
    virtual void visit(const Computed *e) {
    }
    virtual bool is_base_ir_visitor() override {
        return false;
    }
};

class EqSatIRMutator : public IRMutator {
public:
    using IRMutator::visit;
    virtual bool is_base_ir_mutator() override {
        return false;
    }
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

    virtual std::shared_ptr<Var> visit(std::shared_ptr<Var> var) {
        if (var->to_string_var() != nullptr) {
            return var;
        } else if (auto v = var->to_expr_var()) {
            Expr expr = mutate(v->expr);
            if (expr.same_as(v->expr)) {
                return var;
            } else {
                return std::make_shared<ExprVar>(v->loc, expr);
            }
        }
        internal_error << "Unknown Var type\n";
        return {};
    }

    virtual Expr visit(const GLoad *e) {
        Expr predicate = mutate(e->predicate);
        Expr index = mutate(e->index);
        return GLoad::make(e->type, visit(e->name), index, e->image, e->param, predicate, e->alignment);
    }

    virtual Expr visit(const GVariable *e) {
        return GVariable::make(e->type, visit(e->name));
    }

    virtual Expr visit(const Computed *e) {
        return e;
    }

};

}  // namespace EqSatExtensions

/** Rewrite any AMX tile operations that have been stored in the AMXTile memory
 * type as intrinsic calls, to be used in the X86 backend. */
Stmt extract_tile_operations(const Stmt &s);

Stmt eqsat_extract_tile_operations(const Stmt &s);

// EqSat stuffs start here
std::string run_egglog(std::vector<std::pair<std::string, std::string>> &&);

}  // namespace Internal
}  // namespace Halide

#endif
