#include "vert_vert_2d.hpp"
#include <ipc/smooth_contact/primitives/point2.hpp>
#include <ipc/utils/math.hpp>
#include <ipc/distance/point_point.hpp>

namespace ipc {
VertexVertex2D::VertexVertex2D(
    const long& id0,
    const long& id1,
    const CollisionMesh& mesh,
    const double& dhat,
    const Eigen::MatrixXd& V)
    : SmoothCollision<max_vert_2d>(id0, id1, dhat, mesh)
{
    vertices[0] = id0;
    vertices[1] = -1;
    vertices[2] = -1;
    vertices[3] = id1;
    vertices[4] = -1;
    vertices[5] = -1;

    const double dist = point_point_distance(V.row(vertices[0]), V.row(vertices[3]));
    is_active_ = dist < dhat * dhat;

    for (int i_aux = 0; i_aux < 2; i_aux++) {
        const int id = vertices[i_aux * 3];
        if (mesh.vertex_edge_adjacencies()[id].size() != 2)
            logger().error(
                "Invalid number of vertex neighbor in 2D! {} should be 2.",
                mesh.vertex_edge_adjacencies()[id].size());
        for (long i : mesh.vertex_edge_adjacencies()[id]) {
            if (mesh.edges()(i, 0) == id)
                vertices[1 + 3 * i_aux] = mesh.edges()(i, 1);
            else if (mesh.edges()(i, 1) == id)
                vertices[2 + 3 * i_aux] = mesh.edges()(i, 0);
            else
                logger().error("Wrong edge-vertex adjacency!");
        }
    }
}

double VertexVertex2D::operator()(
    const Vector<double, -1, VertexVertex2D::max_size>& positions,
    const ParameterType& params) const
{
    assert(positions.size() == n_dofs());
    const Eigen::Vector2d direc = positions.segment<2>(6) - positions.segment<2>(0);
    const double dist = direc.norm();
    return smooth_point2_term<double>(
               positions.segment<2>(0), direc, positions.segment<2>(2), positions.segment<2>(4), params, true)
        * smooth_point2_term<double>(
               positions.segment<2>(6), -direc, positions.segment<2>(8), positions.segment<2>(10), params, true)
        * Math<double>::inv_barrier(dist / params.dhat, params.r);
}

Vector<double, -1, VertexVertex2D::max_size> VertexVertex2D::gradient(
    const Vector<double, -1, VertexVertex2D::max_size>& positions,
    const ParameterType& params) const
{
    return Eigen::VectorXd::Zero(positions.size());
}

MatrixMax<double, VertexVertex2D::max_size, VertexVertex2D::max_size> VertexVertex2D::hessian(
    const Vector<double, -1, VertexVertex2D::max_size>& positions,
    const ParameterType& params) const
{
    return Eigen::MatrixXd::Zero(positions.size(), positions.size());
}
} // namespace ipc