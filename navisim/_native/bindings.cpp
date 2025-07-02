#include <pybind11/pybind11.h>

namespace py = pybind11;

// Simple C++ function to bind
int add(int a, int b) {
    return a + b;
}

// Define the Python module
PYBIND11_MODULE(renderer, m) {
    m.doc() = "Native bindings for navisim";

    m.def("add", &add, "Add two integers",
          py::arg("a"), py::arg("b"));
}