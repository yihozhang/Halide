#ifndef EQSAT_IR_PARSER_H
#define EQSAT_IR_PARSER_H

#include <string>
#include <vector>
#include "Type.h"
#include "Expr.h"
#include "ExtractTileOperations.h"

namespace Halide {

namespace Internal {

namespace EqSatExtensions {

struct EqSatIRParser {
public:
    EqSatIRParser(const std::string& prog)
        : prog(prog) {};
    Expr parse_expr();
    Expr parse_bop();
    Expr parse_uop();
    std::shared_ptr<Var> parse_var();
    Type parse_type();
    std::string parse_str();
    int parse_int();
    double parse_double();
    std::vector<Expr> parse_vec_expr();
    Stmt parse_stmt();

private:
    std::string prog;
    size_t curr = 0;

    bool is_head(const std::string &head) {
        skip_whitespace();
        if (prog.compare(curr, head.size(), head) == 0) {
            curr += head.size();
            return true;
        } else {
            return false;
        }
    }

    void skip_whitespace() {
        while (curr < prog.size() && isspace(prog[curr])) {
            ++curr;
        }
    }

    void expect(char c) {
        skip_whitespace();
        if (prog[curr] != c) {
            internal_error <<"Expected " << c << " at " << std::to_string(curr);
        }
        ++curr;
    }

    void expect(const std::string &s) {
        if (!is_head(s)) {
            internal_error <<"Expected " << s << " at " << std::to_string(curr) << " but get " << prog.substr(curr);
        }
    }
};

} // namespace EqSatExtensions

}  // namespace Internal

}  // namespace Halide

#endif  // EQSAT_IR_PARSER_H
