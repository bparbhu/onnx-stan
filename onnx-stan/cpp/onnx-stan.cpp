// onnx-stan.cpp
#include <pybind11/pybind11.h>

int square(int x) {
    return x * x;
}

PYBIND11_MODULE(onnx-stan, m) {
    m.def("square", &square, "A function which squares a number");
}
