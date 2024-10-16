auto EGGLOG_PROG = [](std::vector<std::pair<std::string, std::string>>&& bindings) {
    std::string prog;
#include "definition.egg"
#include "analysis/common.egg"
#include "analysis/typechecking.egg"
#include "optimization/vector_axioms.egg"
#include "optimization/arithmetic_axioms.egg"
#include "optimization/constant_folding.egg"
#include "optimization/index_tweaking.egg"
#include "optimization/amx.egg"
    for (auto [name, src] : bindings) {
        prog += "(let " + name + " " + src + ")\n";
    }
    
#include "schedule.egg"

    for (auto [name, _src] : bindings) {
        prog += "(extract " + name + ")\n";
    }
    return prog;
};
