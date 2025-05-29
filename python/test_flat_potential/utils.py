import numpy as np

def heaviside(z, alpha):
    z *= 3 / alpha
    if z < -3:
        return 0
    elif z < -2:
        return (3 + z) ** 3 / 6
    elif z < -1:
        return (((-2 * z - 9) * z - 9) * z + 3) / 6
    elif z < 0:
        return z ** 3 / 6 + 1
    else:
        return 1

def Q(v, w, alpha):
    return heaviside(-v.dot(w) / np.linalg.norm(v) / np.linalg.norm(w), alpha)

def spline(x):
    if x > 1:
        return 0
    else:
        return (1 - x) ** 2

def barrier(d):
    return spline(d) / d

def point_edge_dist_type(p, e0, e1):
    t = (p - e0).dot(e1 - e0) / np.linalg.norm(e1 - e0) ** 2
    if t > 1:
        return 2
    elif t < 0:
        return 0
    else:
        return 1

def point_line_dist(p, e0, e1):
    return abs(np.cross(e0 - p, e1 - p)) / np.linalg.norm(e1 - e0)

def point_edge_dist(p, e0, e1):
    t = (p - e0).dot(e1 - e0) / np.linalg.norm(e1 - e0) ** 2
    if t > 1:
        return np.linalg.norm(p - e1)
    elif t < 0:
        return np.linalg.norm(p - e0)
    else:
        return abs(np.cross(e0 - p, e1 - p)) / np.linalg.norm(e1 - e0)

def point_line_closest_direction(p, e0, e1):
    t = (p - e0).dot(e1 - e0) / np.linalg.norm(e1 - e0) ** 2
    return p - (e0 + t * (e1 - e0))

def point_edge_closest_direction(p, e0, e1):
    t = (p - e0).dot(e1 - e0) / np.linalg.norm(e1 - e0) ** 2
    t = np.clip([t], 0, 1)[0]
    return p - (e0 + t * (e1 - e0))