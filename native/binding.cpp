// navisim_native/binding.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>        // in case you return STL containers
#include "struct/pose.h"

namespace py = pybind11;

PYBIND11_MODULE(pose, m) {
    m.doc() = "Native Pose class for navisim (pybind11)";

    py::class_<Pose>(m, "Pose")
        // constructor
        .def(py::init<
             double, double, double,
             double, double, double>(),
             py::arg("x")=0,   py::arg("y")=0,   py::arg("z")=0,
             py::arg("roll")=0,py::arg("pitch")=0,py::arg("yaw")=0
        )

        // expose fields
        .def_readwrite("x",     &Pose::x)
        .def_readwrite("y",     &Pose::y)
        .def_readwrite("z",     &Pose::z)
        .def_readwrite("roll",  &Pose::roll)
        .def_readwrite("pitch", &Pose::pitch)
        .def_readwrite("yaw",   &Pose::yaw)

        // methods
        .def("translate", &Pose::translate,
             py::arg("dx"), py::arg("dy"), py::arg("dz"),
             "Translate the pose by (dx, dy, dz)")
        .def("rotate", &Pose::rotate,
             py::arg("droll"), py::arg("dpitch"), py::arg("dyaw"),
             "Rotate the pose by (droll, dpitch, dyaw)")
        .def("inverse", &Pose::inverse,
             "Return the inverse of this pose")
        .def("compose", &Pose::compose,
             py::arg("other"),
             "Return this pose composed with another (this ∘ other)")

        // operator* and repr
        .def("__mul__", &Pose::operator*,
             py::arg("other"),
             "Compose two poses with the * operator")
        .def("__repr__", [](const Pose &p) {
            return "<Pose x="   + std::to_string(p.x)
                 + " y="        + std::to_string(p.y)
                 + " z="        + std::to_string(p.z)
                 + " roll="     + std::to_string(p.roll)
                 + " pitch="    + std::to_string(p.pitch)
                 + " yaw="      + std::to_string(p.yaw)
                 + ">";
        })

        // pickle support
        .def(py::pickle(
            // __getstate__: pack the six doubles
            [](Pose const &p) {
                return py::make_tuple(
                    p.x, p.y, p.z,
                    p.roll, p.pitch, p.yaw
                );
            },
            // __setstate__: unpack to recreate a Pose
            [](py::tuple t) {
                if (t.size() != 6)
                    throw std::runtime_error("Invalid Pose state!");
                return Pose(
                    t[0].cast<double>(),
                    t[1].cast<double>(),
                    t[2].cast<double>(),
                    t[3].cast<double>(),
                    t[4].cast<double>(),
                    t[5].cast<double>()
                );
            }
        ));
}