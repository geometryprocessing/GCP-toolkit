#include "edge_vert_2d.hpp"
#include <ipc/smooth_contact/primitives/point2.hpp>
#include <ipc/utils/math.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/point_point.hpp>

namespace ipc {
EdgeVertex2D::EdgeVertex2D(
    const long& id0,
    const long& id1,
    const CollisionMesh& mesh,
    const double& dhat,
    const Eigen::MatrixXd& V)
    : SmoothCollision<max_vert_2d>(id0, id1, dhat, mesh)
{
    vertices[0] = mesh.edges()(id0, 0);
    vertices[1] = mesh.edges()(id0, 1);
    vertices[2] = id1;
    vertices[3] = -1;
    vertices[4] = -1;

    const double dist = point_edge_distance(V.row(vertices[2]), V.row(vertices[0]), V.row(vertices[1]));
    is_active_ = dist < dhat * dhat;

    {
        if (mesh.vertex_edge_adjacencies()[id1].size() != 2)
            logger().error(
                "Invalid number of vertex neighbor in 2D! {} should be 2.",
                mesh.vertex_edge_adjacencies()[id1].size());
        for (long i : mesh.vertex_edge_adjacencies()[id1]) {
            if (mesh.edges()(i, 0) == id1)
                vertices[3] = mesh.edges()(i, 1);
            else if (mesh.edges()(i, 1) == id1)
                vertices[4] = mesh.edges()(i, 0);
            else
                logger().error("Wrong edge-vertex adjacency!");
        }
    }
}

double EdgeVertex2D::operator()(
    const Vector<double, -1, EdgeVertex2D::max_size>& positions,
    const ParameterType& params) const
{
    assert(positions.size() == n_dofs());
    const Eigen::Vector2d x0 = positions.segment<2>(0);
    const Eigen::Vector2d p = positions.segment<2>(4);
    const Eigen::Vector2d e = positions.segment<2>(2) - x0;
    const double l = e.norm();

    Eigen::Vector2d n;
    n << e(1), -e(0);
    n.normalize();

    auto integrand = [&](double theta) {
        const Eigen::Vector2d q = x0 + theta * e;
        const Eigen::Vector2d direc = p - q;
        const double dist = direc.norm();

        return Math<double>::cubic_spline((1. - direc.dot(n) / dist) / params.alpha_n)
            * smooth_point2_term<double>(
                   p, -direc, positions.segment<2>(6), positions.segment<2>(8),
                   params)
            * Math<double>::inv_barrier(dist / params.dhat, params.r);
    };

    // uniform quadrature
    const int n_samples = params.n_quadrature_samples;
    double result = 0.;
    for (int i = 0; i <= n_samples; i++) {
        double f = integrand((double)i / n_samples);
        result += f;
    }
    result *= l / n_samples;

    return result;
}

Vector<double, -1, EdgeVertex2D::max_size> EdgeVertex2D::gradient(
    const Vector<double, -1, EdgeVertex2D::max_size>& positions,
    const ParameterType& params) const
{
    return Eigen::VectorXd::Zero(positions.size());
}

MatrixMax<double, EdgeVertex2D::max_size, EdgeVertex2D::max_size>
EdgeVertex2D::hessian(
    const Vector<double, -1, EdgeVertex2D::max_size>& positions,
    const ParameterType& params) const
{
    return Eigen::MatrixXd::Zero(positions.size(), positions.size());
}
} // namespace ipc