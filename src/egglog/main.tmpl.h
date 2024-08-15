auto EGGLOG_PROG = [](std::string src) {
    std::string prog;
#include "header.egg"
#include "analysis/common.egg"
#include "analysis/typechecking.egg"
#include "optimization/axiom.egg"
#include "optimization/constant_folding.egg"
#include "optimization/amx.egg"
    prog += "(let prog " + src + ")";
#include "schedule.egg"
    return prog;
};
