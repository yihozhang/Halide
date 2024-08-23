#include "EqSatIRPrinter.h"
#include "Error.h"
#include "Expr.h"

#define GENERATE_VISIT_BINOP(NAME)                                               \
    void EqSatIRPrinter::visit(const NAME *e) {                                  \
        printArgs(stream, "Bop", "(" #NAME ")", EXPR_ARG(e->a), EXPR_ARG(e->b)); \
    }

#define TYPE_ARG(type) ([this, e]() { print_type(type); })
#define EXPR_ARG(expr) ([this, e]() { (expr).accept(this); })
#define NAME_ARG(name) ("\"" + (name) + "\"")
namespace Halide {

namespace Internal {

void EqSatIRPrinter::print_type(const Type &t) const {
    auto code = t.code();
    std::string s;
    switch (code) {
    case halide_type_int:
        s = "Int";
        break;
    case halide_type_uint:
        s = "UInt";
        break;
    case halide_type_float:
        s = "Float";
        break;
    case halide_type_bfloat:
        s = "BFloat";
        break;
    case halide_type_handle:
        s = "Handle";
        break;
    default:
        internal_error << "Unknown Halide type code: " << code << "\n";
    }

    stream << "(";
    if (t.is_handle()) {
        stream << s;
    } else {
        stream << s << " " << t.bits();
    }
    stream << " " << t.lanes() << ")";
}

// Helper to check if a type is callable with std::ostream&
template<typename T, typename = void>
struct is_callable_with_ostream : std::false_type {};

// Specialization for callable types with std::ostream&
template<typename T>
struct is_callable_with_ostream<T, std::void_t<decltype(std::declval<T>()())>> : std::true_type {};

// Function to print a callable object with std::ostream& or a direct value
template<typename T>
void printValue(std::ostream &s, const T &value) {
    if constexpr (is_callable_with_ostream<T>::value) {
        value();
    } else {
        s << value;
    }
}

// Base case: no arguments left to process
void printArgsImpl(std::ostream &s) {
    s << ")";
}

// Recursive case: handle one argument and process remaining arguments
template<typename T, typename... Args>
void printArgsImpl(std::ostream &s, const T &firstArg, const Args &...remainingArgs) {
    s << " ";
    printValue(s, firstArg);
    printArgsImpl(s, remainingArgs...);
}
template<typename T, typename... Args>
void printArgs(std::ostream &s, const T &firstArg, const Args &...remainingArgs) {
    s << "(";
    printValue(s, firstArg);
    printArgsImpl(s, remainingArgs...);
}

void EqSatIRPrinter::visit(const IntImm *e) {
    printArgs(stream, "IntImm", e->type.bits(), e->value);
}

void EqSatIRPrinter::visit(const UIntImm *e) {
    printArgs(stream, "UIntImm", e->type.bits(), e->value);
}

void EqSatIRPrinter::visit(const FloatImm *e) {
    auto value = std::to_string(e->value).find('.') != std::string::npos ? std::to_string(e->value) : std::to_string(e->value) + ".0";
    printArgs(stream, "FloatImm", e->type.bits(), value);
}

void EqSatIRPrinter::visit(const StringImm *e) {
    user_error << "Not supported\n";
}

void EqSatIRPrinter::visit(const Cast *e) {
    printArgs(stream, "Cast", TYPE_ARG(e->type), EXPR_ARG(e->value));
}

void EqSatIRPrinter::visit(const Reinterpret *e) {
    printArgs(stream, "Reinterpret", TYPE_ARG(e->type), EXPR_ARG(e->value));
}

void EqSatIRPrinter::visit(const Variable *e) {
    printArgs(stream, "Var", TYPE_ARG(e->type), NAME_ARG(e->name));
}

GENERATE_VISIT_BINOP(Add)
GENERATE_VISIT_BINOP(Sub)
GENERATE_VISIT_BINOP(Mul)
GENERATE_VISIT_BINOP(Div)
GENERATE_VISIT_BINOP(Mod)
GENERATE_VISIT_BINOP(Min)
GENERATE_VISIT_BINOP(Max)
GENERATE_VISIT_BINOP(EQ)
GENERATE_VISIT_BINOP(NE)
GENERATE_VISIT_BINOP(LT)
GENERATE_VISIT_BINOP(LE)
GENERATE_VISIT_BINOP(GT)
GENERATE_VISIT_BINOP(GE)
GENERATE_VISIT_BINOP(And)
GENERATE_VISIT_BINOP(Or)

void EqSatIRPrinter::visit(const Not *e) {
    printArgs(stream, "Uop", "(Not)", EXPR_ARG(e->a));
}

void EqSatIRPrinter::visit(const Select *e) {
    printArgs(stream, "Select", EXPR_ARG(e->condition), EXPR_ARG(e->true_value), EXPR_ARG(e->false_value));
}

void EqSatIRPrinter::visit(const Load *e) {
    printArgs(stream, "Load", TYPE_ARG(e->type), NAME_ARG(e->name), EXPR_ARG(e->index));
}

void EqSatIRPrinter::visit(const Ramp *e) {
    printArgs(stream, "Ramp", EXPR_ARG(e->base), EXPR_ARG(e->stride), e->lanes);
}

void EqSatIRPrinter::visit(const Broadcast *e) {
    printArgs(stream, "Broadcast", EXPR_ARG(e->value), e->lanes);
}

void EqSatIRPrinter::visit(const Call *e) {
    // (Call fn type (vec-of args...))
    stream << "(Call " << e->name << " ";
    print_type(e->type);
    stream << " (vec-of";

    for (const auto &arg : e->args) {
        stream << " ";
        arg.accept(this);
    }
    stream << "))";
}

void EqSatIRPrinter::visit(const Let *e) {
    user_error << "Not supported\n";
}

void EqSatIRPrinter::visit(const VectorReduce *e) {
    internal_assert(e->op == VectorReduce::Operator::Add) << "Only supporting Add for now\n";
    printArgs(stream, "VectorReduce", TYPE_ARG(e->type), "(Add)", EXPR_ARG(e->value));
}

void EqSatIRPrinter::visit(const MemToAMX *e) {
    printArgs(stream, "Mem2AMX", EXPR_ARG(e->expr));
}

void EqSatIRPrinter::visit(const AMXToMem *e) {
    printArgs(stream, "AMX2Mem", EXPR_ARG(e->expr));
}

void EqSatIRPrinter::visit(const Store *e) {
    printArgs(stream, "Store", NAME_ARG(e->name), EXPR_ARG(e->index), EXPR_ARG(e->value));
}

void EqSatIRPrinter::print(const Expr &e) {
    e.accept(this);
}

void EqSatIRPrinter::print(const Stmt &e) {
    e.accept(this);
}

}  // namespace Internal
}  // namespace Halide