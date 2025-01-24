#include "Halide.h"

using namespace Halide;

template<typename T>
void fill_buffer_flat(Buffer<T> &buf, int row, int acc) {
    for (int iy = 0; iy < row; ++iy) {
        for (int ix = 0; ix < acc; ++ix) {
            T val = rand() % 2;
            buf(ix, iy) = val;
        }
    }
}

template<typename T>
void fill_buffer_flat_one(Buffer<T> &buf, int row, int acc) {
    for (int iy = 0; iy < row; ++iy) {
        for (int ix = 0; ix < acc; ++ix) {
            T val = 1;
            buf(ix, iy) = val;
        }
    }
}
