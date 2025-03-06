#pragma once

#include <ipc/collision_mesh.hpp>
#include "smooth_collisions.hpp"
#include <ipc/utils/MaybeParallelFor.hpp>
#include <ipc/smooth_contact/high_order/edge_edge_2d.hpp>
#include <ipc/smooth_contact/high_order/edge_vert_2d.hpp>
#include <ipc/smooth_contact/high_order/vert_vert_2d.hpp>

#include <Eigen/Core>

namespace ipc {

template <int dim> class SmoothCollisionsBuilder;

template <> class SmoothCollisionsBuilder<2> {
public:
    SmoothCollisionsBuilder() { }

    void add_edge_vertex_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<EdgeVertexCandidate>& candidates,
        const ParameterType& param,
        const std::function<double(const long&)>& vert_dhat,
        const std::function<double(const long&)>& edge_dhat,
        const size_t start_i,
        const size_t end_i);

    // -------------------------------------------------------------------------

    static void merge(
        const utils::ParallelCacheType<SmoothCollisionsBuilder<2>>&
            local_storage,
        SmoothCollisions<2>& merged_collisions);

    // Constructed collisions
    std::vector<std::shared_ptr<typename SmoothCollisions<2>::value_type>>
        collisions;

    // -------------------------------------------------------------------------

    // Store the indices to pairs to avoid duplicates.
    unordered_map<
        std::pair<long, long>,
        std::shared_ptr<SmoothCollisionTemplate<max_vert_2d, Point2, Point2>>>
        vert_vert_2_to_id;
    unordered_map<
        std::pair<long, long>,
        std::shared_ptr<SmoothCollisionTemplate<max_vert_2d, Edge2, Point2>>>
        vert_edge_2_to_id;
};

template <> class SmoothCollisionsBuilder<3> {
public:
    SmoothCollisionsBuilder() { }

    void add_edge_edge_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<EdgeEdgeCandidate>& candidates,
        const ParameterType& param,
        const std::function<double(const long&)>& vert_dhat,
        const std::function<double(const long&)>& edge_dhat,
        const size_t start_i,
        const size_t end_i);

    void add_face_vertex_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<FaceVertexCandidate>& candidates,
        const ParameterType& param,
        const std::function<double(const long&)>& vert_dhat,
        const std::function<double(const long&)>& edge_dhat,
        const std::function<double(const long&)>& face_dhat,
        const size_t start_i,
        const size_t end_i);

    // -------------------------------------------------------------------------

    static void merge(
        const utils::ParallelCacheType<SmoothCollisionsBuilder<3>>&
            local_storage,
        SmoothCollisions<3>& merged_collisions);

    // Constructed collisions
    std::vector<std::shared_ptr<typename SmoothCollisions<3>::value_type>>
        collisions;

    // -------------------------------------------------------------------------

    // Store the indices to pairs to avoid duplicates, no need for Face-Vertex
    // and Edge-Edge
    unordered_map<
        std::pair<long, long>,
        std::shared_ptr<SmoothCollisionTemplate<max_vert_3d, Point3, Point3>>>
        vert_vert_3_to_id;
    unordered_map<
        std::pair<long, long>,
        std::shared_ptr<SmoothCollisionTemplate<max_vert_3d, Edge3, Point3>>>
        edge_vert_3_to_id;
};

template <int dim> class HighOrderSmoothCollisionsBuilder;

template <> class HighOrderSmoothCollisionsBuilder<2> {
public:
    HighOrderSmoothCollisionsBuilder() { }

    void add_edge_vertex_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<EdgeVertexCandidate>& candidates,
        const ParameterType& param,
        const std::function<double(const long&)>& vert_dhat,
        const std::function<double(const long&)>& edge_dhat,
        const size_t start_i,
        const size_t end_i);

    // -------------------------------------------------------------------------

    static void merge(
        const utils::ParallelCacheType<HighOrderSmoothCollisionsBuilder<2>>&
            local_storage,
        SmoothCollisions<2>& merged_collisions);

    // Constructed collisions
    std::vector<std::shared_ptr<typename SmoothCollisions<2>::value_type>>
        collisions;

    // -------------------------------------------------------------------------

    // Store the indices to pairs to avoid duplicates.
    unordered_map<std::pair<long, long>, std::shared_ptr<VertexVertex2D>>
        vert_vert_2_to_id;
    unordered_map<std::pair<long, long>, std::shared_ptr<EdgeVertex2D>>
        edge_vert_2_to_id;
    unordered_map<std::pair<long, long>, std::shared_ptr<EdgeEdge2D>>
        edge_edge_2_to_id;
};

template <> class HighOrderSmoothCollisionsBuilder<3> {
public:
    HighOrderSmoothCollisionsBuilder() { }

    void add_edge_edge_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<EdgeEdgeCandidate>& candidates,
        const ParameterType& param,
        const std::function<double(const long&)>& vert_dhat,
        const std::function<double(const long&)>& edge_dhat,
        const size_t start_i,
        const size_t end_i) {}

    void add_face_vertex_collisions(
        const CollisionMesh& mesh,
        const Eigen::MatrixXd& vertices,
        const std::vector<FaceVertexCandidate>& candidates,
        const ParameterType& param,
        const std::function<double(const long&)>& vert_dhat,
        const std::function<double(const long&)>& edge_dhat,
        const std::function<double(const long&)>& face_dhat,
        const size_t start_i,
        const size_t end_i) {}

    // -------------------------------------------------------------------------

    static void merge(
        const utils::ParallelCacheType<HighOrderSmoothCollisionsBuilder<3>>&
            local_storage,
        SmoothCollisions<3>& merged_collisions) {}

    // Constructed collisions
    std::vector<std::shared_ptr<typename SmoothCollisions<3>::value_type>>
        collisions;

    // -------------------------------------------------------------------------

    // Store the indices to pairs to avoid duplicates.
    // unordered_map<std::pair<long, long>, std::shared_ptr<VertexVertex3D>>
    //     vert_vert_3_to_id;
    // unordered_map<std::pair<long, long>, std::shared_ptr<EdgeVertex3D>>
    //     edge_vert_3_to_id;
    // unordered_map<std::pair<long, long>, std::shared_ptr<EdgeEdge3D>>
    //     edge_edge_3_to_id;
    // unordered_map<std::pair<long, long>, std::shared_ptr<VertexVertex3D>>
    //     vert_vert_3_to_id;
    // unordered_map<std::pair<long, long>, std::shared_ptr<EdgeVertex3D>>
    //     edge_vert_3_to_id;
    // unordered_map<std::pair<long, long>, std::shared_ptr<EdgeEdge3D>>
    //     edge_edge_3_to_id;
};
} // namespace ipc