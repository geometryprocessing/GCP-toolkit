#pragma once
#include <cmath>
#include <ipc/utils/logger.hpp>

namespace ipc {

constexpr static int n_vert_neighbors_2d = 3;
constexpr static int n_edge_neighbors_2d = 2;
constexpr static int max_vert_2d =
    2 * std::max(n_vert_neighbors_2d, n_edge_neighbors_2d);
constexpr static int n_vert_neighbors_3d = 20; // increase me if needed
constexpr static int n_edge_neighbors_3d = 4;
constexpr static int n_face_neighbors_3d = 3;
constexpr static int max_vert_3d = n_vert_neighbors_3d * 2;

template <int dim> class MaxVertices;
template <> class MaxVertices<2> { public: static constexpr int value = max_vert_2d; };
template <> class MaxVertices<3> { public: static constexpr int value = max_vert_3d; };

struct ParameterType {
    ParameterType(
        const double _dhat,
        const double _alpha_t,
        const double _beta_t,
        const int _r):
        ParameterType(_dhat, _alpha_t, _beta_t, 0, 0.1, _r, 1000) {}
    ParameterType(
        const double _dhat,
        const double _alpha_t,
        const double _beta_t,
        const double _alpha_n,
        const double _beta_n,
        const int _r,
        const int _n_quadrature_samples)
        : dhat(_dhat)
        , alpha_t(_alpha_t)
        , beta_t(_beta_t)
        , alpha_n(_alpha_n)
        , beta_n(_beta_n)
        , r(_r)
        , n_quadrature_samples(_n_quadrature_samples)
    {
        if (!(r > 0) || !(dhat > 0) || 
        !(abs(alpha_t) <= 1) || !(abs(alpha_n) <= 1) || 
        !(abs(beta_t) <= 1) || !(abs(beta_n) <= 1) || 
        !(beta_t + alpha_t > 1e-6) || !(beta_n + alpha_n > 1e-6) || (n_quadrature_samples < 1))
            logger().error(
                "Wrong parameters for smooth contact! dhat {} alpha_t {} beta_t {} alpha_n {} beta_n {} r {}",
                dhat, alpha_t, beta_t, alpha_n, beta_n, r);
    }
    ParameterType() {}

    void set_adaptive_dhat_ratio(const double adaptive_dhat_ratio_)
    {
        adaptive_dhat_ratio = adaptive_dhat_ratio_;
    }
    double get_adaptive_dhat_ratio() const { return adaptive_dhat_ratio; }

    double dhat = 1;
    double alpha_t = 1, beta_t = 0;
    double alpha_n = 0.1, beta_n = 0;
    int r = 2;
    int n_quadrature_samples = 1000;

private:
    double adaptive_dhat_ratio = 0.5;
};

} // namespace ipc