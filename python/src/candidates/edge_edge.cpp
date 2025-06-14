#include <common.hpp>

#include <ipc/candidates/edge_edge.hpp>

namespace py = pybind11;
using namespace ipc;

void define_edge_edge_candidate(py::module_& m)
{
    py::class_<EdgeEdgeCandidate, CollisionStencil<4>>(m, "EdgeEdgeCandidate")
        .def(
            py::init<index_t, index_t>(), py::arg("edge0_id"),
            py::arg("edge1_id"))
        .def(
            py::init([](std::tuple<index_t, index_t> edge_ids) {
                return std::make_unique<EdgeEdgeCandidate>(
                    std::get<0>(edge_ids), std::get<1>(edge_ids));
            }),
            py::arg("edge_ids"))
        .def("known_dtype", &EdgeEdgeCandidate::known_dtype)
        .def(
            "__str__",
            [](const EdgeEdgeCandidate& ee) {
                return fmt::format("[{:d}, {:d}]", ee.edge0_id, ee.edge1_id);
            })
        .def(
            "__repr__",
            [](const EdgeEdgeCandidate& ee) {
                return fmt::format(
                    "EdgeEdgeCandidate({:d}, {:d})", ee.edge0_id, ee.edge1_id);
            })
        .def("__eq__", &EdgeEdgeCandidate::operator==, py::arg("other"))
        .def("__ne__", &EdgeEdgeCandidate::operator!=, py::arg("other"))
        .def(
            "__lt__", &EdgeEdgeCandidate::operator<,
            "Compare EdgeEdgeCandidates for sorting.", py::arg("other"))
        .def_readwrite(
            "edge0_id", &EdgeEdgeCandidate::edge0_id, "ID of the first edge.")
        .def_readwrite(
            "edge1_id", &EdgeEdgeCandidate::edge1_id, "ID of the second edge.");

    py::implicitly_convertible<
        std::tuple<index_t, index_t>, EdgeEdgeCandidate>();
}
