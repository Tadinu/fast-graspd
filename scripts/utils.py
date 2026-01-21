import numpy as np
import warp as wp


# ---------------------------------------------------------------------------- #
#  Utility functions (host side – NumPy)                                        #
# ---------------------------------------------------------------------------- #
def rotation_matrix_between(vec_src: np.ndarray, vec_dst: np.ndarray) -> np.ndarray:
    """
    Return the 3×3 rotation matrix that rotates vec_src onto vec_dst.

    Both arguments must be 1-D arrays with 3 elements.
    """
    a = (vec_src / np.linalg.norm(vec_src)).reshape(3)
    b = (vec_dst / np.linalg.norm(vec_dst)).reshape(3)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    kmat = np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]],
        dtype=np.float64,
    )

    return np.eye(3) + kmat + kmat.dot(kmat) * ((1.0 - c) / (s ** 2))


# ---------------------------------------------------------------------------- #
#  Warp helper functions                                                        #
# ---------------------------------------------------------------------------- #
@wp.func
def project_to_plane(p: wp.vec3, c: wp.vec3, n: wp.vec3) -> wp.vec3:
    """Project point p onto the plane with point c and normal n."""
    return p - wp.dot(p - c, n) * n


@wp.func
def gravity_vector_for_dir(dir_idx: int) -> wp.vec3:
    """Return the (scaled) gravity vector corresponding to dir_idx."""
    g = wp.vec3(0.0, 0.0, 0.0)

    if dir_idx == 1:
        g = wp.vec3(1.0, 0.0, 0.0)
    elif dir_idx == 2:
        g = wp.vec3(-1.0, 0.0, 0.0)
    elif dir_idx == 3:
        g = wp.vec3(0.0, 1.0, 0.0)
    elif dir_idx == 4:
        g = wp.vec3(0.0, -1.0, 0.0)
    elif dir_idx == 5:
        g = wp.vec3(0.0, 0.0, 1.0)
    elif dir_idx == 6:
        g = wp.vec3(0.0, 0.0, -1.0)

    return g


# -------------------------------------------------------------------------- #
#  Rotational displacement helper                                            #
# -------------------------------------------------------------------------- #
@wp.func
def rot_disp_for_dir(dir_idx: int) -> wp.vec3:
    """Return the rotational displacement (axis–angle) for *dir_idx*.

    The original implementation considered only seven translational
    directions (0 = none, ±x, ±y, ±z).  We extend this to a total of
    thirteen directions by appending positive and negative rotations
    about each principal axis.  The mapping is as follows::

        0  : no perturbation
        1  : +x translation     7  : +x rotation
        2  : −x translation     8  : −x rotation
        3  : +y translation     9  : +y rotation
        4  : −y translation    10  : −y rotation
        5  : +z translation    11  : +z rotation
        6  : −z translation    12  : −z rotation

    For translational directions the returned vector is zero.  For the
    rotational directions we return an axis-angle vector of magnitude
    one radian about the respective axis; this is subsequently scaled
    by *dt* inside the kernel to obtain a small angular displacement
    consistent with the translation scaling (translation of 1 unit
    → 1 rad rotation).
    """

    r = wp.vec3(0.0, 0.0, 0.0)

    # ±X rotation
    if dir_idx == 7:
        r = wp.vec3(1.0, 0.0, 0.0)
    elif dir_idx == 8:
        r = wp.vec3(-1.0, 0.0, 0.0)

    # ±Y rotation
    elif dir_idx == 9:
        r = wp.vec3(0.0, 1.0, 0.0)
    elif dir_idx == 10:
        r = wp.vec3(0.0, -1.0, 0.0)

    # ±Z rotation
    elif dir_idx == 11:
        r = wp.vec3(0.0, 0.0, 1.0)
    elif dir_idx == 12:
        r = wp.vec3(0.0, 0.0, -1.0)

    return r


# ────────────────────────────────────────────────────────────────
#  Forward function
# ────────────────────────────────────────────────────────────────
@wp.func
def leaky_max(a: float, b: float, r: float) -> float:
    """
    Leaky max:

        if a > b:  return a
        else:      return b            (same as regular max)

    but in the backward pass we leak a fraction *r* of the
    negative gradient to *a* when a ≤ b.
    """
    if a > b:
        return a
    return b


# ────────────────────────────────────────────────────────────────
#  Custom gradient (adjoint)
# ────────────────────────────────────────────────────────────────
@wp.func_grad(leaky_max)
def adj_leaky_max(a: float, b: float, r: float, adj_ret: float):
    """
    Asymmetric gradient you specified:

        if a > b:           ∂L/∂a += adj_ret
        else:
            if adj_ret < 0: ∂L/∂a += r * adj_ret
            ∂L/∂b += adj_ret
        (no gradient to r)
    """
    if a > b:
        wp.adjoint[a] += adj_ret
    else:
        if adj_ret < 0.0:
            wp.adjoint[a] += r * adj_ret
        wp.adjoint[b] += adj_ret
