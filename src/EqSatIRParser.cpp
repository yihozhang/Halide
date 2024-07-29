#include "EqSatIRParser.h"
#include "IR.h"
#include "IROperator.h"

#define BOP_CASE(op) \
    if (is_head(#op)) { \
        expect(')'); \
        auto lhs = parse_expr(); \
        auto rhs = parse_expr(); \
        return op::make(lhs, rhs); \
    }

namespace Halide {

namespace Internal {

Expr EqSatIRParser::parse_expr() {
    expect('(');

    Expr result;
    // Expr
    if (is_head("Cast")) {
        auto type = parse_type();
        auto expr = parse_expr();
        result = Cast::make(type, expr);
    } else if (is_head("ReInterpret")) {
        auto type = parse_type();
        auto expr = parse_expr();
        result = Reinterpret::make(type, expr);
    } else if (is_head("Bop")) {
        result = parse_bop();
    } else if (is_head("Uop")) {
        auto child = parse_expr();
        auto lhs = parse_expr();
        // TODO
        result = Expr();
    } else if (is_head("Select")) {
        auto pred = parse_expr();
        auto true_val = parse_expr();
        auto false_val = parse_expr();
        result = Select::make(pred, true_val, false_val);
    } else if (is_head("Load")) {
        auto type = parse_type();
        auto name = parse_str();
        auto index = parse_expr();
        result = Load::make(type, name, index, Buffer<>(), Parameter(), const_true(), ModulusRemainder());
    } else if (is_head("Ramp")) {
        auto base = parse_expr();
        auto stride = parse_expr();
        auto lanes = parse_int();
        result = Ramp::make(base, stride, lanes);
    } else if (is_head("Broadcast")) {
        auto value = parse_expr();
        auto lanes = parse_int();
        result = Broadcast::make(value, lanes);
    } else if (is_head("Let")) {
        auto var = parse_str();
        auto value = parse_expr();
        auto body = parse_expr();
        result = Let::make(var, value, body);
    } else if (is_head("Call")) {
        auto name = parse_str();
        auto type = parse_type();
        auto args = parse_vec_expr();
        result = Call::make(type, name, args, Call::Intrinsic);
    } else if (is_head("Var")) {
        auto type = parse_type();
        auto name = parse_str();
        result = Variable::make(type, name);
    } else if (is_head("IntImm")) {
        auto bits = parse_int();
        auto value = parse_int();
        result = IntImm::make(Int(bits), value);
    } else if (is_head("UIntImm")) {
        auto bits = parse_int();
        auto value = parse_int();
        result = UIntImm::make(UInt(bits), value);
    } else if (is_head("FloatImm")) {
        auto bits = parse_int();
        auto value = parse_double();
        result = FloatImm::make(Float(bits), value);
    } else {
        internal_error << "Unknown expression at " << std::to_string(curr);
    }
    expect(')');
    return result;
}

Expr EqSatIRParser::parse_bop() {
    expect('(');
    BOP_CASE(Add);
    BOP_CASE(Sub);
    BOP_CASE(Mul);
    BOP_CASE(Div);
    BOP_CASE(Mod);
    BOP_CASE(Min);
    BOP_CASE(Max);
    BOP_CASE(EQ);
    BOP_CASE(NE);
    BOP_CASE(LT);
    BOP_CASE(LE);
    BOP_CASE(GT);
    BOP_CASE(GE);
    BOP_CASE(And);
    BOP_CASE(Or);
    internal_error << "Unknown binary operator at " << std::to_string(curr);
    return Expr();
}

Expr EqSatIRParser::parse_uop() {
    expect('(');
    // not is the only unary operator currently
    if (is_head("Not")) {
        expect(')');
        auto child = parse_expr();
        return Not::make(child);
    }
    internal_error << "Unknown unary operator at " << std::to_string(curr);
    return Expr();
}

Type EqSatIRParser::parse_type() {
    expect('(');
    Type result;
    if (is_head("Handle")) {
        auto bits = parse_int();
        result = Handle(bits);
    } else {
        std::vector<std::pair<std::string, halide_type_code_t>> head({{"Int", halide_type_int},
                                                                      {"UInt", halide_type_uint},
                                                                      {"Float", halide_type_float},
                                                                      {"BFloat", halide_type_bfloat}});
        for (auto &h : head) {
            if (is_head(h.first)) {
                auto bits = parse_int();
                auto lanes = parse_int();
                result = Type().with_bits(bits).with_lanes(lanes).with_code(h.second);
                break;
            }
        }
    }
    expect(')');
    return result;
}

std::string EqSatIRParser::parse_str() {
    expect('"');
    std::string result;
    while (prog[curr] != '"') {
        result += prog[curr];
        ++curr;
    }
    ++curr;
    return result;
}

int EqSatIRParser::parse_int() {
    skip_whitespace();
    return std::stoi(prog.substr(curr), &curr);
}

double EqSatIRParser::parse_double() {
    skip_whitespace();
    return std::stod(prog.substr(curr), &curr);
}

std::vector<Expr> EqSatIRParser::parse_vec_expr() {
    expect('(');
    expect("vec-of");
    std::vector<Expr> result;
    while (!is_head(")")) {
        result.push_back(parse_expr());
    }
    return result;
}

}  // namespace Internal

}  // namespace Halide