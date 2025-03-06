#pragma once

#include <ipc/collision_mesh.hpp>
#include <ipc/smooth_contact/collisions/smooth_collision.hpp>

namespace ipc {
class VertexVertex2D : public SmoothCollision<max_vert_2d> {
public:
    VertexVertex2D(
        const long& id0,
        const long& id1,
        const CollisionMesh& mesh,
        const double& dhat,
        const Eigen::MatrixXd& V);

    std::string name() const override { return "vert-vert"; }
    inline int num_vertices() const override { return 6; }
    inline int n_dofs() const override { return num_vertices() * 2; }
    CollisionType type() const override { return CollisionType::VertexVertex; }

    // ---- non distance type potential ----

    double operator()(
        const Vector<double, -1, max_size>& positions,
        const ParameterType& params) const override;

    Vector<double, -1, max_size> gradient(
        const Vector<double, -1, max_size>& positions,
        const ParameterType& params) const override;

    MatrixMax<double, max_size, max_size> hessian(
        const Vector<double, -1, max_size>& positions,
        const ParameterType& params) const override;

    // ---- distance ----

    double compute_distance(
        const Vector<double, -1, max_size>& positions) const override
    {
        return 0.;
    }

    Vector<double, -1, max_size> compute_distance_gradient(
        const Vector<double, -1, max_size>& positions) const override
    {
        return Vector<double, -1, max_size>::Zero(positions.size());
    }

    MatrixMax<double, max_size, max_size> compute_distance_hessian(
        const Vector<double, -1, max_size>& positions) const override
    {
        return MatrixMax<double, max_size, max_size>::Zero(positions.size(), positions.size());
    }
};
} // namespace ipc
