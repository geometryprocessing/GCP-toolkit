#pragma once

#include <ipc/smooth_contact/distance/mollifier.hpp>
#include "primitive.hpp"

namespace ipc {
class Face : public Primitive {
public:
    constexpr static int n_core_points = 3;
    constexpr static int dim = 3;
    // d is a vector from closest point on the face to the point outside of the
    // face
    Face(
        const long& id,
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const VectorMax3d& d,
        const ParameterType& param);

    int n_vertices() const override;
    int n_dofs() const override { return n_vertices() * dim; }

    double potential(const Vector3d& d, const Vector9d& x) const;
    Vector12d grad(const Vector3d& d, const Vector9d& x) const;
    Matrix12d hessian(const Vector3d& d, const Vector9d& x) const;
};

/// @brief d points from triangle to the point
template <typename scalar>
scalar smooth_face_term(
    const Eigen::Ref<const Vector3<scalar>>& v0,
    const Eigen::Ref<const Vector3<scalar>>& v1,
    const Eigen::Ref<const Vector3<scalar>>& v2,
    const Eigen::Ref<const Vector3<scalar>>& dn)
{
    // return 0.5 * (v1 - v0).cross(v2 - v0).norm(); // area of triangle
    Vector3<scalar> n = (v2 - v0).cross(v1 - v0);
    return pow(dn.dot(n) / n.norm(), 6);
}
} // namespace ipc