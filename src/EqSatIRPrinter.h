#ifndef EQSAT_IR_PRINTER_H
#define EQSAT_IR_PRINTER_H

#include <iostream>
#include "IRVisitor.h"
#include "Module.h"
#include "Scope.h"
#include "IRPrinter.h"

namespace Halide {
namespace Internal {

class EqSatIRPrinter : public IRVisitor {
public:
    /** Construct an IRPrinter pointed at a given output stream
     * (e.g. std::cout, or a std::ofstream) */
    explicit EqSatIRPrinter(std::ostream &stream) : stream(stream) {}

    /** emit an expression on the output stream */
    void print(const Expr &);

    // /** emit a statement on the output stream */
    // void print(const Stmt &);

    // static void test();

protected:
    /** The stream on which we're outputting */
    std::ostream &stream;
    void print_type(const Type& t);
    void visit(const IntImm *) override;
    void visit(const UIntImm *) override;
    void visit(const FloatImm *) override;
    void visit(const StringImm *) override;
    void visit(const Cast *) override;
    void visit(const Reinterpret *) override;
    void visit(const Variable *) override;
    void visit(const Add *) override;
    void visit(const Sub *) override;
    void visit(const Mul *) override;
    void visit(const Div *) override;
    void visit(const Mod *) override;
    void visit(const Min *) override;
    void visit(const Max *) override;
    void visit(const EQ *) override;
    void visit(const NE *) override;
    void visit(const LT *) override;
    void visit(const LE *) override;
    void visit(const GT *) override;
    void visit(const GE *) override;
    void visit(const And *) override;
    void visit(const Or *) override;
    void visit(const Not *) override;
    void visit(const Select *) override;
    void visit(const Load *) override;
    void visit(const Ramp *) override;
    void visit(const Broadcast *) override;
    void visit(const Call *) override;
    void visit(const Let *) override;
};

}  // namespace Internal
}  // namespace Halide


#endif
