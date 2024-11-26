auto EGGLOG_PROG = [](std::vector<std::pair<std::string, std::string>>&& bindings) {
    std::string prog;
#include "definition.egg"
#include "analysis/common.egg"
#include "analysis/typechecking.egg"
#include "optimization/vector_axioms.egg"
#include "optimization/arithmetic_axioms.egg"
#include "optimization/constant_folding.egg"
#include "optimization/index_tweaking.egg"
#include "accelerator/amx.egg"
#include "accelerator/cuda_wmma.egg"
    for (auto [name, src] : bindings) {
        prog += "(let " + name + " " + src + ")\n";
    }
    
#include "schedule.egg"

    std::string input_names;
    for (auto [name, _src] : bindings) {
        prog += "(extract " + name + ")\n";
        input_names += '\"' + name + '\"' + " ";
    }
    prog = std::regex_replace(prog, std::regex("__INPUTS__"), input_names);
    return prog;
};
