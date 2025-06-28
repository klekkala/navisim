#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>

// Simple function
int add(int a, int b) {
    return a + b;
}

// Function with STL containers
std::vector<int> multiply_vector(const std::vector<int>& input, int multiplier) {
    std::vector<int> result;
    for (int val : input) {
        result.push_back(val * multiplier);
    }
    return result;
}

// A simple class
class Calculator {
public:
    Calculator(double initial_value) : value(initial_value) {}
    
    void add(double x) { value += x; }
    void multiply(double x) { value *= x; }
    double get_value() const { return value; }
    void reset() { value = 0.0; }
    
    // Method that returns a string
    std::string to_string() const {
        return "Calculator value: " + std::to_string(value);
    }

private:
    double value;
};

// Pybind11 module definition
PYBIND11_MODULE(math_operations, m) {
    m.doc() = "pybind11 example plugin";
    
    // Bind functions
    m.def("add", &add, "A function that adds two numbers");
    m.def("multiply_vector", &multiply_vector, "Multiply each element in vector by a value");
    
    // Bind class
    pybind11::class_<Calculator>(m, "Calculator")
        .def(pybind11::init<double>())
        .def("add", &Calculator::add)
        .def("multiply", &Calculator::multiply)
        .def("get_value", &Calculator::get_value)
        .def("reset", &Calculator::reset)
        .def("to_string", &Calculator::to_string)
        .def("__repr__", &Calculator::to_string);
}