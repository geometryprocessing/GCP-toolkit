#include "face.hpp"
#include <ipc/utils/AutodiffTypes.hpp>

namespace ipc {
// namespace {
//     template <typename T>
//     std::array<Vector3<T>, 3> double_to_autodiff(
//         const Eigen::Ref<const Eigen::Vector3d>& v0,
//         const Eigen::Ref<const Eigen::Vector3d>& v1,
//         const Eigen::Ref<const Eigen::Vector3d>& v2)
//     {
//         Vector3<T> v0_, v1_, v2_;
//         for (int d = 0; d < 3; d++)
//         {
//             v0_(d) = T(  d, v0(d));
//             v1_(d) = T(3+d, v1(d));
//             v2_(d) = T(6+d, v2(d));
//         }
//         return {{v0_, v1_, v2_}};
//     }
// }
Face::Face(
    const long& id,
    const CollisionMesh& mesh,
    const Eigen::MatrixXd& vertices,
    const VectorMax3d& d,
    const ParameterType& param)
    : Primitive(id, param)
{
    _vert_ids = { { mesh.faces()(id, 0), mesh.faces()(id, 1),
                    mesh.faces()(id, 2) } };
    // Vector3d a = vertices.row(_vert_ids[1]) - vertices.row(_vert_ids[0]);
    // Vector3d b = vertices.row(_vert_ids[2]) - vertices.row(_vert_ids[0]);
    // is_active_ = a.cross(b).dot(d) > 0;
    is_active_ = true;
}
int Face::n_vertices() const { return n_face_neighbors_3d; }
double Face::potential(const Vector3d& d, const Vector9d& x) const
{
    return smooth_face_term<double>(x.head<3>(), x.segment<3>(3), x.tail<3>(), d);
}
Vector12d Face::grad(const Vector3d& d, const Vector9d& x) const
{
    Vector12d g;
    g.setZero();
    DiffScalarBase::setVariableCount(12);
    auto X = slice_positions<ADGrad<12>, 4, 3>((Vector12d() << d, x).finished());
    g =
        smooth_face_term<ADGrad<12>>(X.row(1), X.row(2), X.row(3), X.row(0)).getGradient();
    return g;
}
Matrix12d Face::hessian(const Vector3d& d, const Vector9d& x) const
{
    Matrix12d h;
    h.setZero();
    DiffScalarBase::setVariableCount(12);
    auto X = slice_positions<ADHessian<12>, 4, 3>((Vector12d() << d, x).finished());
    h =
        smooth_face_term<ADHessian<12>>(X.row(1), X.row(2), X.row(3), X.row(0))
            .getHessian();
    return h;
}
} // namespace ipc
