// navisim_native/binding.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>        // for STL containers
#include "struct/pose.h"

namespace py = pybind11;

PYBIND11_MODULE(pose, m) {
    m.doc() = "Native Pose class for navisim (pybind11)";

    py::class_<Pose>(m, "Pose")
        // constructor: now takes an optional timestamp
        .def(py::init<
             double, double, double,
             double, double, double,
             std::int64_t>(),
             py::arg("x")=0,    py::arg("y")=0,    py::arg("z")=0,
             py::arg("roll")=0, py::arg("pitch")=0, py::arg("yaw")=0,
             py::arg("timestamp_ns")=Pose::now_ns()
        )

        // expose fields, including timestamp_ns
        .def_readwrite("x",            &Pose::x)
        .def_readwrite("y",            &Pose::y)
        .def_readwrite("z",            &Pose::z)
        .def_readwrite("roll",         &Pose::roll)
        .def_readwrite("pitch",        &Pose::pitch)
        .def_readwrite("yaw",          &Pose::yaw)
        .def_readwrite("timestamp_ns", &Pose::timestamp_ns,
                       "Nanoseconds since Unix epoch when this pose was created/last updated")

        // methods
        .def("translate", &Pose::translate,
             py::arg("dx"), py::arg("dy"), py::arg("dz"),
             "Translate the pose and update timestamp")
        .def("rotate", &Pose::rotate,
             py::arg("droll"), py::arg("dpitch"), py::arg("dyaw"),
             "Rotate the pose and update timestamp")
        .def("inverse", &Pose::inverse,
             "Return the inverse of this pose (carries original timestamp)")
        .def("compose", &Pose::compose,
             py::arg("other"),
             "Return this pose composed with another (timestamp = max of both)")

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
                 + " ts_ns="    + std::to_string(p.timestamp_ns)
                 + ">";
        })
        .def(py::pickle(
            // __getstate__: pack all seven fields
            [](Pose const &p) {
                return py::make_tuple(
                    p.x, p.y, p.z,
                    p.roll, p.pitch, p.yaw,
                    p.timestamp_ns
                );
            },
            // __setstate__: unpack to recreate a Pose
            [](py::tuple t) {
                if (t.size() != 7)
                    throw std::runtime_error("Invalid Pose state!");
                return Pose(
                    t[0].cast<double>(),
                    t[1].cast<double>(),
                    t[2].cast<double>(),
                    t[3].cast<double>(),
                    t[4].cast<double>(),
                    t[5].cast<double>(),
                    t[6].cast<std::int64_t>()
                );
            }
        ));
}