#include "edge_edge_2d.hpp"
#include <ipc/utils/math.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/edge_edge.hpp>

namespace ipc {
EdgeEdge2D::EdgeEdge2D(
    const long& id0,
    const long& id1,
    const CollisionMesh& mesh,
    const double& dhat,
    const Eigen::MatrixXd& V)
    : SmoothCollision<max_vert_2d>(id0, id1, dhat, mesh)
{
    vertices[0] = mesh.edges()(id0, 0);
    vertices[1] = mesh.edges()(id0, 1);
    vertices[2] = mesh.edges()(id1, 0);
    vertices[3] = mesh.edges()(id1, 1);

    Eigen::Matrix<double, 4, 3> P;
    P.setZero();
    P.leftCols<2>() << V.row(vertices[0]), V.row(vertices[1]),
        V.row(vertices[2]), V.row(vertices[3]);

    const double dist =
        edge_edge_distance(P.row(0), P.row(1), P.row(2), P.row(3));

    is_active_ = dist < dhat * dhat;
}

double EdgeEdge2D::operator()(
    const Vector<double, -1, EdgeEdge2D::max_size>& positions,
    const ParameterType& params) const
{
    assert(positions.size() == n_dofs());
    const Eigen::Vector2d x0 = positions.segment<2>(0);
    const Eigen::Vector2d x1 = positions.segment<2>(4);
    const Eigen::Vector2d e0 = positions.segment<2>(2) - x0;
    const Eigen::Vector2d e1 = positions.segment<2>(6) - x1;
    const double l0 = e0.norm();
    const double l1 = e1.norm();

    Eigen::Vector2d n0, n1;
    n0 << e0(1), -e0(0);
    n1 << e1(1), -e1(0);
    n0.normalize();
    n1.normalize();

    auto integrand = [&](double theta0, double theta1) {
        const Eigen::Vector2d p0 = x0 + theta0 * e0;
        const Eigen::Vector2d p1 = x1 + theta1 * e1;
        const Eigen::Vector2d d = p1 - p0;
        const double dist = d.norm();

        return Math<double>::smooth_heaviside(
                   d.dot(n0) / dist - 1., params.alpha_n, params.beta_n)
            * Math<double>::smooth_heaviside(
                   -d.dot(n1) / dist - 1., params.alpha_n, params.beta_n)
            * Math<double>::inv_barrier(dist / params.dhat, params.r);
    };

    // uniform quadrature
    const int n_samples = params.n_quadrature_samples;
    double result = 0.;
    for (int i = 0; i <= n_samples; i++) {
        for (int j = 0; j <= n_samples; j++) {
            double f = integrand(static_cast<double>(i) / n_samples, static_cast<double>(j) / n_samples);
            result += f;
        }
    }
    result *= (l0 * l1) / (n_samples * n_samples);

    return result;
}

Vector<double, -1, EdgeEdge2D::max_size> EdgeEdge2D::gradient(
    const Vector<double, -1, EdgeEdge2D::max_size>& positions,
    const ParameterType& params) const
{
    return Eigen::VectorXd::Zero(positions.size());
}

MatrixMax<double, EdgeEdge2D::max_size, EdgeEdge2D::max_size>
EdgeEdge2D::hessian(
    const Vector<double, -1, EdgeEdge2D::max_size>& positions,
    const ParameterType& params) const
{
    return Eigen::MatrixXd::Zero(positions.size(), positions.size());
}
} // namespace ipc