"""
https://github.com/dylanturpin/fast-graspd
https://journals.sagepub.com/doi/10.1177/0278364909105606
https://projects.csail.mit.edu/manipulation/rss07/paper__dexterous_grasping_via_eigengrasps_a_low_dimensional_approach_to_a_high_complexity_problem__ciocarlie.pdf
Grasp collection and optimisation script for the Allegro right-hand gripper.

This module searches for collision-free, stable grasps on a target object mesh.
object mesh.  A differentiable Warp simulation is used to evaluate contact
forces and object motion, while simple SGD updates optimise the hand's joint
configuration.  Optionally, the script can export USD files that visualise the
final grasp and the full optimisation trajectory.
"""

# ---------------------------------------------------------------------------- #
#  Imports                                                                     #
# ---------------------------------------------------------------------------- #
import os
import math
import uuid
from pathlib import Path

import hydra
import numpy as np
import trimesh as tri
import warp.context
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

import newton
import warp as wp
from newton_backend import NewtonBackend
from utils import rotation_matrix_between, rot_disp_for_dir, project_to_plane, gravity_vector_for_dir, leaky_max

# ---------------------------------------------------------------------------- #
wp.init()
# wp.config.cache_kernels = False
# wp.config.print_launches = True
# wp.config.verbose = False

WP_GRAVITY_ZERO = wp.zeros(1, dtype=wp.vec3)
# gravity in cm/s^2
WP_GRAVITY_EARTH = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)


# ---------------------------------------------------------------------------- #
#  Warp kernels                                                                 #
# ---------------------------------------------------------------------------- #

@wp.kernel
def wp_kernel_copy_joint_q_for_render(
        # inputs
        src_joint_q: wp.array(dtype=float, ndim=1),  # N_BATCH · N_JOINTS
        batch_idx: wp.int32,
        # sizes
        NUM_JOINTS: wp.int32,
        # outputs
        dst_joint_q: wp.array(dtype=float, ndim=1),  # N_JOINTS
):
    """
    Copy joint configuration for one batch element to the render model.
    """
    tid = wp.tid()
    src_offset = batch_idx * NUM_JOINTS
    dst_joint_q[tid] = src_joint_q[src_offset + tid]


# -------------------------------------------------------------------------- #
#  Reset per-iteration buffers, now including rotational state               #
# -------------------------------------------------------------------------- #
@wp.kernel
def wp_reset_buffers(
        obj_q: wp.array(dtype=wp.vec3, ndim=2),  # N_BATCH × N_DIRS (translation)
        obj_qd: wp.array(dtype=wp.vec3, ndim=2),  # N_BATCH × N_DIRS (linear vel.)
        obj_ang: wp.array(dtype=wp.vec3, ndim=2),  # rotational displacement (axis-angle)
        obj_angd: wp.array(dtype=wp.vec3, ndim=2),  # angular velocity
        loss: wp.array(dtype=float, ndim=1),
        loss_base: wp.array(dtype=float, ndim=1),
        loss_hand_pose_l2_reg: wp.array(dtype=float, ndim=1),
        loss_self_interp: wp.array(dtype=float, ndim=1),
        loss_hand_obj_interp: wp.array(dtype=float, ndim=1),
        loss_hand_pose_lower: wp.array(dtype=float, ndim=1),
        loss_hand_pose_upper: wp.array(dtype=float, ndim=1),
        # sizes
        NUM_DIRS: wp.int32,
):
    """
    Zero-out per-iteration buffers.
    """
    tid = wp.tid()
    batch_idx = tid // NUM_DIRS
    dir_idx = tid % NUM_DIRS

    obj_q[batch_idx, dir_idx] = wp.vec3()
    obj_qd[batch_idx, dir_idx] = wp.vec3()
    obj_ang[batch_idx, dir_idx] = wp.vec3()
    obj_angd[batch_idx, dir_idx] = wp.vec3()

    # Only once per launch
    if tid == 0:
        loss[0] = 0.0
        loss_base[0] = 0.0
        loss_hand_pose_l2_reg[0] = 0.0
        loss_self_interp[0] = 0.0
        loss_hand_obj_interp[0] = 0.0
        loss_hand_pose_lower[0] = 0.0
        loss_hand_pose_upper[0] = 0.0


# -------------------------------------------------------------------------- #
#  Contact response kernel (translation + rotation)                          #
# -------------------------------------------------------------------------- #
@wp.kernel
def wp_kernel_contact_step_kernel(
        # inputs
        obj_mesh_id: wp.uint64,
        obj_mesh_vert_normals: wp.array(dtype=wp.vec3, ndim=1),
        body_q: wp.array(dtype=wp.transform, ndim=1),
        contact_shape0: wp.array(dtype=wp.int32, ndim=1),
        shape_body: wp.array(dtype=wp.int32, ndim=1),
        contact_point0: wp.array(dtype=wp.vec3, ndim=1),
        dt: float,
        padding: wp.array(dtype=float, ndim=1),
        w_hand_obj_interp: float,
        MU: float,
        # sizes
        NUM_DIRS: wp.int32,
        NUM_CONTACTS: wp.int32,
        # outputs (aggregated per-direction)
        obj_q: wp.array(dtype=wp.vec3, ndim=2),  # translation
        obj_qd: wp.array(dtype=wp.vec3, ndim=2),  # linear velocity
        obj_ang: wp.array(dtype=wp.vec3, ndim=2),  # rotation (axis-angle)
        obj_angd: wp.array(dtype=wp.vec3, ndim=2),  # angular velocity
        loss_hand_obj_interp: wp.array(dtype=float, ndim=1),
        # constant inputs for torque calc
        obj_com: wp.vec3,
        inertia_scalar: float,
):
    """
    Compute contact response for each (batch, direction, contact) tuple.
    """

    # --------------------------------------------------------------------- #
    #  Thread indexing                                                      #
    # --------------------------------------------------------------------- #
    tid = wp.tid()
    batch_idx = tid // (NUM_DIRS * NUM_CONTACTS)
    dir_idx = (tid % (NUM_DIRS * NUM_CONTACTS)) // NUM_CONTACTS
    contact_idx = tid % NUM_CONTACTS

    # --------------------------------------------------------------------- #
    #  Directional perturbations (translation + rotation)                   #
    # --------------------------------------------------------------------- #
    trans_disp = gravity_vector_for_dir(dir_idx)
    rot_disp = rot_disp_for_dir(dir_idx)

    # --------------------------------------------------------------------- #
    #  World-space contact point                                            #
    # --------------------------------------------------------------------- #
    model_contact_idx = batch_idx * NUM_CONTACTS + contact_idx

    shape_idx = contact_shape0[model_contact_idx]
    body_idx = shape_body[shape_idx]

    contact_x = wp.transform_point(
        body_q[body_idx], contact_point0[model_contact_idx]
    )

    # Unconstrained motion (translation)
    obj_qd_unconstrained = trans_disp * dt
    obj_q_unconstrained = obj_qd_unconstrained * dt

    # Unconstrained motion (rotation)
    obj_angd_unconstrained = rot_disp * dt
    obj_ang_unconstrained = obj_angd_unconstrained * dt

    # --------------------------------------------------------------------- #
    #  Mesh query                                                           #
    # --------------------------------------------------------------------- #
    max_dist = wp.float32(1e8)
    query = wp.mesh_query_point(obj_mesh_id, contact_x, max_dist)
    face_index = query.face
    face_u = query.u
    face_v = query.v
    face_w = 1.0 - face_u - face_v
    sign = query.sign

    i0 = wp.mesh_get_index(obj_mesh_id, face_index * 3 + 0)
    i1 = wp.mesh_get_index(obj_mesh_id, face_index * 3 + 1)
    i2 = wp.mesh_get_index(obj_mesh_id, face_index * 3 + 2)

    p0 = wp.mesh_get_point(obj_mesh_id, face_index * 3 + 0)
    p1 = wp.mesh_get_point(obj_mesh_id, face_index * 3 + 1)
    p2 = wp.mesh_get_point(obj_mesh_id, face_index * 3 + 2)

    n0 = obj_mesh_vert_normals[i0]
    n1 = obj_mesh_vert_normals[i1]
    n2 = obj_mesh_vert_normals[i2]

    # Barycentric interpolation
    p = face_u * p0 + face_v * p1 + face_w * p2

    c0 = project_to_plane(p, p0, n0)
    c1 = project_to_plane(p, p1, n1)
    c2 = project_to_plane(p, p2, n2)
    q = face_u * c0 + face_v * c1 + face_w * c2

    alpha = 0.75
    r = (1.0 - alpha) * p + alpha * q
    closest_pt = r + obj_q_unconstrained
    # wp.printf("%.8f %.8f %.8f\n", obj_q_unconstrained[0], obj_q_unconstrained[1], obj_q_unconstrained[2])

    # ------------------------------------------------------------------ #
    #  Side of plane test                                                #
    # ------------------------------------------------------------------ #
    interp_n = wp.normalize(face_u * n0 + face_v * n1 + face_w * n2)
    dot_prod = wp.dot(interp_n, contact_x - r)

    if dot_prod > 0.0:
        sign = 1.0
    else:
        sign = -1.0

    # Signed distance plus padding
    if sign < 0.0:
        d = -wp.length(contact_x - closest_pt) - padding[0]
    else:
        d = wp.length(contact_x - closest_pt) - padding[0]

    # ------------------------------------------------------------------ #
    #  Contact impulse                                                   #
    # ------------------------------------------------------------------ #
    if d < -0.01:
        wp.atomic_add(loss_hand_obj_interp, 0, w_hand_obj_interp * d * d)

    C = 0.0
    d_inv = -d  # positive when penetration
    C = leaky_max(d_inv, 0.0, 0.1)

    if sign < 0.0:
        n = -wp.normalize(contact_x - closest_pt)
    else:
        n = wp.normalize(contact_x - closest_pt)

    hand_delta_q = n
    obj_delta_q = -hand_delta_q

    eps = 1e-5
    lmbda = (-C) / (wp.length(obj_delta_q) * wp.length(obj_delta_q) + eps)

    # ------------------------------------------------------------------ #
    #  Coulomb friction (tangential impulse)                             #
    # ------------------------------------------------------------------ #
    # Tangential component of unconstrained velocity
    t_vec = obj_qd_unconstrained - wp.dot(obj_qd_unconstrained, n) * n
    t_len = wp.length(t_vec)

    if t_len > 1.0e-8:
        t_hat = t_vec / t_len
        # Friction impulse magnitude limited by µ * |normal_impulse|
        fric_lambda = -MU * wp.abs(lmbda)
        fric_delta_q = fric_lambda * t_hat

        # Corresponding torque / angular displacement
        r_vec_fric = contact_x - obj_com
        fric_ang_delta = wp.cross(r_vec_fric, fric_delta_q) / inertia_scalar
        fric_ang_vel = fric_ang_delta / dt

        wp.atomic_add(
            obj_q, batch_idx, dir_idx, fric_delta_q / float(NUM_CONTACTS)
        )
        wp.atomic_add(
            obj_qd, batch_idx, dir_idx, fric_delta_q / dt / float(NUM_CONTACTS)
        )
        wp.atomic_add(
            obj_ang, batch_idx, dir_idx, fric_ang_delta / float(NUM_CONTACTS)
        )
        wp.atomic_add(
            obj_angd, batch_idx, dir_idx, fric_ang_vel / float(NUM_CONTACTS)
        )

    # ------------------------------------------------------------------ #
    #  Compute torque / angular displacement                              #
    # ------------------------------------------------------------------ #

    # Vector from COM to contact point (approx)
    r_vec = contact_x - obj_com
    # Impulse approximation (displacement over dt):
    imp = lmbda * obj_delta_q  # same units as displacement; scalar factor ok

    ang_delta = wp.cross(r_vec, imp) / inertia_scalar  # axis-angle approx (small rot)
    # Angular velocity
    ang_vel = ang_delta / dt

    # ------------------------------------------------------------------ #
    #  Accumulate object displacements (trans + rot)                      #
    # ------------------------------------------------------------------ #
    if contact_idx == 0:
        wp.atomic_add(obj_q, batch_idx, dir_idx, obj_q_unconstrained)
        wp.atomic_add(obj_qd, batch_idx, dir_idx, obj_qd_unconstrained)
        wp.atomic_add(obj_ang, batch_idx, dir_idx, obj_ang_unconstrained)
        wp.atomic_add(obj_angd, batch_idx, dir_idx, obj_angd_unconstrained)

    # used to encourage "coverage" of every translation direction
    if dir_idx > 0 and dir_idx < 7:
        delta_proj = wp.dot(lmbda * obj_delta_q, wp.normalize(trans_disp))
        if delta_proj <= 0.0:
            wp.atomic_add(
                obj_q,
                batch_idx,
                dir_idx,
                delta_proj * trans_disp / float(NUM_CONTACTS)
            )
            wp.atomic_add(
                obj_qd,
                batch_idx,
                dir_idx,
                delta_proj * trans_disp / dt / float(NUM_CONTACTS)
            )
            # rotational part scaled similarly (no directional preference)
            wp.atomic_add(
                obj_ang,
                batch_idx,
                dir_idx,
                ang_delta / float(NUM_CONTACTS)
            )
            wp.atomic_add(
                obj_angd,
                batch_idx,
                dir_idx,
                ang_vel / float(NUM_CONTACTS)
            )
    else:
        wp.atomic_add(
            obj_q, batch_idx, dir_idx, lmbda * obj_delta_q / float(NUM_CONTACTS)
        )
        wp.atomic_add(
            obj_qd,
            batch_idx,
            dir_idx,
            lmbda * obj_delta_q / dt / float(NUM_CONTACTS),
        )
        wp.atomic_add(
            obj_ang,
            batch_idx,
            dir_idx,
            ang_delta / float(NUM_CONTACTS)
        )
        wp.atomic_add(
            obj_angd,
            batch_idx,
            dir_idx,
            ang_vel / float(NUM_CONTACTS)
        )


@wp.kernel
def wp_kernel_compute_total_loss(
        # inputs
        joint_q: wp.array(dtype=float, ndim=1),  # N_BATCH · N_JOINTS
        joint_limit_lower: wp.array(dtype=float, ndim=1),
        joint_limit_upper: wp.array(dtype=float, ndim=1),
        obj_q: wp.array(dtype=wp.vec3, ndim=2),
        obj_qd: wp.array(dtype=wp.vec3, ndim=2),
        obj_angd: wp.array(dtype=wp.vec3, ndim=2),
        loss_self_interp: wp.array(dtype=float, ndim=1),
        loss_hand_obj_interp: wp.array(dtype=float, ndim=1),
        w_base: float,
        w_l2_mid: float,
        w_limit: float,
        obj_radius: float,
        # sizes
        NUM_BATCH: wp.int32,
        NUM_DIRS: wp.int32,
        NUM_JOINTS: wp.int32,
        # outputs
        loss: wp.array(dtype=float, ndim=1),
        loss_base: wp.array(dtype=float, ndim=1),
        loss_hand_pose_l2_reg: wp.array(dtype=float, ndim=1),
        loss_hand_pose_lower: wp.array(dtype=float, ndim=1),
        loss_hand_pose_upper: wp.array(dtype=float, ndim=1),
):
    """
    Aggregate all loss terms: hand-object interaction, joint regularisation,
    and object motion.
    """
    tid = wp.tid()

    # ------------------------------------------------------------------ #
    #  Loss terms independent of thread                                  #
    # ------------------------------------------------------------------ #
    if tid == 0:
        wp.atomic_add(loss, 0, loss_self_interp[0])
        wp.atomic_add(loss, 0, loss_hand_obj_interp[0])

    # ------------------------------------------------------------------ #
    #  Joint-based regularisation                                        #
    # ------------------------------------------------------------------ #
    batch_idx = tid // (NUM_JOINTS - 7)
    joint_local_idx = tid % (NUM_JOINTS - 7) + 7  # skip base joints

    if batch_idx < NUM_BATCH:
        joint_global_idx = batch_idx * NUM_JOINTS + joint_local_idx
        j_val = joint_q[joint_global_idx]

        limit_offset = batch_idx * (NUM_JOINTS - 7) + (joint_local_idx - 7)
        lower = joint_limit_lower[limit_offset]
        upper = joint_limit_upper[limit_offset]

        # Keep joint near the middle of its span
        from_mid = j_val - (lower + upper) * 0.5
        l_mid = w_l2_mid * from_mid * from_mid
        wp.atomic_add(loss_hand_pose_l2_reg, 0, l_mid)
        wp.atomic_add(loss, 0, l_mid)

        # Penalise limit violations
        under_lower = wp.max(lower - j_val, 0.0)
        l_lower = w_limit * under_lower * under_lower
        wp.atomic_add(loss_hand_pose_lower, 0, l_lower)
        wp.atomic_add(loss, 0, l_lower)

        over_upper = wp.max(j_val - upper, 0.0)
        l_upper = w_limit * over_upper * over_upper
        wp.atomic_add(loss_hand_pose_upper, 0, l_upper)
        wp.atomic_add(loss, 0, l_upper)

    # ------------------------------------------------------------------ #
    #  Object motion loss                                                #
    # ------------------------------------------------------------------ #
    batch_idx_obj = tid // NUM_DIRS
    dir_idx_obj = tid % NUM_DIRS

    if batch_idx_obj < NUM_BATCH:
        speed = wp.length(obj_qd[batch_idx_obj, dir_idx_obj])
        rot_speed = wp.length(obj_angd[batch_idx_obj, dir_idx_obj])

        # Convert angular speed to linear speed at object surface
        lin_equiv_rot = rot_speed * obj_radius

        l = w_base * speed * speed + w_base * lin_equiv_rot * lin_equiv_rot

        wp.atomic_add(loss, 0, l)
        wp.atomic_add(loss_base, 0, l)


@wp.kernel
def wp_kernel_apply_grad_step(
        hand_q_grad: wp.array(dtype=float, ndim=1),  # N_BATCH · N_JOINTS
        lr: float,
        # cyclic_idx: wp.int32,
        # sizes
        NUM_JOINTS: wp.int32,
        # outputs
        hand_q: wp.array(dtype=float, ndim=1),
):
    """
    Apply SGD-style update step with simple per-component learning-rates.
    """
    tid = wp.tid()
    batch_idx = tid // (NUM_JOINTS - 5)
    joint_local_idx = tid % (NUM_JOINTS - 5)

    offset = batch_idx * NUM_JOINTS

    # Base translation --------------------------------------------------- #
    if joint_local_idx == 0:  # and joint_local_idx == cyclic_idx:
        wp.atomic_sub(hand_q, offset + 0, 1e-3 * lr * hand_q_grad[offset + 0])
        wp.atomic_sub(hand_q, offset + 1, 1e-3 * lr * hand_q_grad[offset + 1])
        wp.atomic_sub(hand_q, offset + 2, 1e-3 * lr * hand_q_grad[offset + 2])

    # Base orientation --------------------------------------------------- #
    elif joint_local_idx == 1:  # and joint_local_idx == cyclic_idx:
        wp.atomic_sub(hand_q, offset + 3, 1e-2 * lr * hand_q_grad[offset + 3])
        wp.atomic_sub(hand_q, offset + 4, 1e-2 * lr * hand_q_grad[offset + 4])
        wp.atomic_sub(hand_q, offset + 5, 1e-2 * lr * hand_q_grad[offset + 5])
        wp.atomic_sub(hand_q, offset + 6, 1e-2 * lr * hand_q_grad[offset + 6])

        # Re-normalise quaternion
        q = wp.quat(
            hand_q[offset + 3],
            hand_q[offset + 4],
            hand_q[offset + 5],
            hand_q[offset + 6],
        )
        q = wp.normalize(q)
        hand_q[offset + 3] = q[0]
        hand_q[offset + 4] = q[1]
        hand_q[offset + 5] = q[2]
        hand_q[offset + 6] = q[3]

    # Articulated joints ------------------------------------------------- #
    # elif joint_local_idx == cyclic_idx:
    else:
        joint_idx = offset + joint_local_idx + 5
        wp.atomic_sub(hand_q, joint_idx, lr * hand_q_grad[joint_idx])


# ---------------------------------------------------------------------------- #
#  Main optimisation loop                                                       #
# ---------------------------------------------------------------------------- #
cfg = None


@hydra.main(config_path="../conf/collect_grasps", config_name="config")
def get_collect_grasps_config(cfg_: DictConfig) -> None:
    global cfg
    cfg = cfg_


get_collect_grasps_config()
# ------------------------------------------------------------------ #
#  Global constants / hyper-parameters (config-driven)               #
# ------------------------------------------------------------------ #
# All parameters are configurable via Hydra; the defaults reproduce
# the original behaviour and work with existing configuration files.

NUM_DIRS: int = getattr(cfg.collector_config, "num_dirs", 7)
NUM_BATCH: int = getattr(cfg.collector_config, "batch_size", 4)
NUM_ITERS: int = getattr(cfg.collector_config, "num_iters", 15_001)

dt: float = getattr(cfg.collector_config, "dt", 1e-2)
lr: float = getattr(cfg.collector_config, "lr", 2e-4)

# Number of random initialisations evaluated per object
NUM_BATCH_PER_OBJ: int = getattr(cfg.collector_config, "num_batch_per_obj", 64)

# Output / rendering ------------------------------------------------ #
obj_set = cfg.collector_config.obj_set
hand_name = cfg.collector_config.hand_name

# Whether to export USDs of the final grasp and/or optimisation trajectory
RENDER_FINAL: bool = getattr(cfg.collector_config, "render_final_grasp", True)
RENDER_TRAJ: bool = getattr(cfg.collector_config, "render_opt_traj", False)

# Base directory for outputs (per-object sub-folders are appended)
base_output_dir: str = getattr(cfg.collector_config, "output_dir", "grasp_outputs")

# ------------------------------------------------------------------ #
#  Loss weighting parameters                                        #
# ------------------------------------------------------------------ #
w_base: float = getattr(cfg.collector_config, "w_base_loss", 1e4)
w_hand_obj_interp: float = getattr(cfg.collector_config, "w_hand_obj_interp_loss", 1e1)
w_l2_mid: float = getattr(cfg.collector_config, "w_l2_mid_loss", 1.0)
w_limit: float = getattr(cfg.collector_config, "w_limit_loss", 1e2)
# Friction coefficient (Coulomb)
mu_friction: float = getattr(cfg.collector_config, "friction_mu", 0.5)

# ------------------------------------------------------------------ #
#  Build simulation model                                            #
# ------------------------------------------------------------------ #
builder = newton.ModelBuilder()
gripper_urdf_path = to_absolute_path(
    os.path.join(
        os.path.dirname(__file__),
        "../assets/grippers/allegro_right_simplified/"
        "allegro_right_resampled_contacts.urdf",
    )
)

# Load batch
for _ in range(NUM_BATCH):
    builder.add_urdf(
        gripper_urdf_path,
        xform=wp.transform(
            wp.vec3(0.0, 0.0, 0.0),
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5),
        ),
        floating=True,
        # density=0.1,
        # armature=0.1,
        # stiffness=0.0,
        # damping=0.0,
        # contact_ke=1e4,
        # contact_kd=1e2,
        # contact_kf=1e2,
        # contact_mu=1.0,
        # limit_ke=1e4,
        enable_self_collisions=False,
        parse_visuals_as_colliders=True,
        # limit_kd=1e1,
    )

NUM_JOINTS = len(builder.joint_q) // NUM_BATCH
# builder.joint_count != len(builder.joint_q)

# Finalise model ---------------------------------------------------- #
builder.shape_collision_filter_pairs = {
    (i, j) for i in range(builder.shape_count) for j in range(builder.shape_count)
}

model = builder.finalize("cuda")
model.requires_grad = True
model.ground = True
model.joint_attach_ke = 1600.0
model.joint_attach_kd = 20.0

state = model.state()
contacts = model.collide(state)

NUM_CONTACTS = len(contacts.rigid_contact_shape0) // NUM_BATCH

model.joint_q.requires_grad = True
state.body_q.requires_grad = True

# ------------------------------------------------------------------ #
#  Load target object mesh                                           #
# ------------------------------------------------------------------ #
obj_name = cfg.collector_config.obj_name

obj_path = os.path.join("../assets/", obj_set, f"{obj_name}.obj")
if obj_set == "ycb-original-meshes":
    obj_path = os.path.join(
        "assets/", obj_set, obj_name, "google_16k", "textured.obj"
    )
if obj_set == "ycb-tetwild":
    obj_path = f"/assets/ycb-tetwild/{obj_name}.ply"

obj_mesh_tri = tri.load_mesh(to_absolute_path(obj_path))

# Warp representations --------------------------------------------- #
joint_limit_lower = wp.array(
    data=np.asarray(builder.joint_limit_lower, dtype=np.float32),
    dtype=float,
    device="cuda",
    requires_grad=True,
)
joint_limit_upper = wp.array(
    data=np.asarray(builder.joint_limit_upper, dtype=np.float32),
    dtype=float,
    device="cuda",
    requires_grad=True,
)
obj_mesh_verts = wp.array(
    data=obj_mesh_tri.vertices,
    dtype=wp.vec3,
    device="cuda",
    requires_grad=True,
)
obj_mesh_inds = wp.array(
    data=obj_mesh_tri.faces.flatten(),
    dtype=int,
    device="cuda",
    requires_grad=True,
)
obj_mesh_vert_normals = wp.array(
    data=obj_mesh_tri.vertex_normals,
    dtype=wp.vec3,
    device="cuda",
    requires_grad=True,
)

obj_mesh = wp.Mesh(obj_mesh_verts, obj_mesh_inds)
obj_mesh.refit()
obj_meshes = {"obj": obj_mesh}

# ------------------------------------------------------------------ #
#  Compute object centre-of-mass and an approximate inertia scalar    #
# ------------------------------------------------------------------ #
obj_com_np: np.ndarray = obj_mesh_tri.vertices.mean(axis=0)
obj_com_vec = wp.vec3(float(obj_com_np[0]), float(obj_com_np[1]), float(obj_com_np[2]))

# Simple bounding-sphere approximation of the moment of inertia.      #
# This is sufficient for distributing impulse between translation     #
# and rotation in a grasp-planning context.                           #
radius = np.linalg.norm(obj_mesh_tri.vertices - obj_com_np, axis=1).max()
inertia_scalar = float((2.0 / 5.0) * radius * radius)  # m=1 assumed

obj_mesh_id = obj_mesh.id

# ------------------------------------------------------------------ #
#  Allocate optimisation buffers                                     #
# ------------------------------------------------------------------ #
obj_q = wp.zeros((NUM_BATCH, NUM_DIRS), dtype=wp.vec3, device="cuda", requires_grad=True)
obj_qd = wp.zeros_like(obj_q)

# ------------------------------------------------------------------ #
#  New buffers for rotational motion (orientation/ang. velocity)      #
# ------------------------------------------------------------------ #
# We represent small orientation changes as an axis-angle vector      #
# capturing the integrated angular displacement for each optimisation
# step.  The corresponding time-derivative stores angular velocity.    #
obj_ang = wp.zeros((NUM_BATCH, NUM_DIRS), dtype=wp.vec3, device="cuda", requires_grad=True)
obj_angd = wp.zeros_like(obj_ang, requires_grad=True)

loss = wp.zeros(1, dtype=float, device="cuda", requires_grad=True)
loss_base = wp.zeros_like(loss)
loss_hand_pose_l2_reg = wp.zeros_like(loss)
loss_hand_pose_upper = wp.zeros_like(loss)
loss_hand_pose_lower = wp.zeros_like(loss)
loss_self_interp = wp.zeros_like(loss)
loss_hand_obj_interp = wp.zeros_like(loss)

padding = wp.zeros(1, dtype=float, device="cuda", requires_grad=True)

# ------------------------------------------------------------------ #
#  Build grasp optimizing graph                                      #
# ------------------------------------------------------------------ #
tape = wp.Tape()
with wp.ScopedCapture(device="cuda") as grasp_optimizing_capture:
    with tape:
        # Reset buffers
        wp.launch(
            kernel=wp_reset_buffers,
            dim=NUM_BATCH * NUM_DIRS,
            inputs=[
                obj_q,
                obj_qd,
                obj_ang,
                obj_angd,
                loss,
                loss_base,
                loss_hand_pose_l2_reg,
                loss_self_interp,
                loss_hand_obj_interp,
                loss_hand_pose_lower,
                loss_hand_pose_upper,
                NUM_DIRS,
            ],
            device="cuda",
        )

        # Forward kinematics
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

        # Contact step
        wp.launch(
            kernel=wp_kernel_contact_step_kernel,
            dim=NUM_DIRS * NUM_BATCH * NUM_CONTACTS,
            inputs=[
                obj_mesh_id,
                obj_mesh_vert_normals,
                state.body_q,
                contacts.rigid_contact_shape0,
                model.shape_body,
                contacts.rigid_contact_point0,
                dt,
                padding,
                w_hand_obj_interp,
                mu_friction,
                NUM_DIRS,
                NUM_CONTACTS,
                obj_q,
                obj_qd,
                obj_ang,
                obj_angd,
                loss_hand_obj_interp,
                obj_com_vec,
                inertia_scalar,
            ],
            device="cuda",
        )

        # Loss aggregation
        wp.launch(
            kernel=wp_kernel_compute_total_loss,
            dim=max(NUM_BATCH * NUM_DIRS, NUM_BATCH * (NUM_JOINTS - 7)),
            inputs=[
                model.joint_q,
                joint_limit_lower,
                joint_limit_upper,
                obj_q,
                obj_qd,
                obj_angd,
                loss_self_interp,
                loss_hand_obj_interp,
                w_base,
                w_l2_mid,
                w_limit,
                radius,
                NUM_BATCH,
                NUM_DIRS,
                NUM_JOINTS,
                loss,
                loss_base,
                loss_hand_pose_l2_reg,
                loss_hand_pose_lower,
                loss_hand_pose_upper,
            ],
            device="cuda",
        )
    tape.backward(loss)
grasp_optimizing_graph = grasp_optimizing_capture.graph

# ------------------------------------------------------------------ #
#  Renderer setup                                                    #
# ------------------------------------------------------------------ #
render_model_builder = newton.ModelBuilder()
render_model_urdf_path = to_absolute_path(
    os.path.join(
        os.path.dirname(__file__),
        "../assets/grippers/allegro_right_simplified/allegro_right_display.urdf",
    )
)
render_model_builder.add_urdf(
    render_model_urdf_path,
    xform=wp.transform(
        wp.vec3(0.0, 0.0, 0.0),
        wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5),
    ),
    floating=True,
    # density=0.1,
    # armature=0.1,
    # stiffness=0.0,
    # damping=0.0,
    # contact_ke=1e4,
    # contact_kd=1e2,
    # contact_kf=1e2,
    # contact_mu=1.0,
    # limit_ke=1e4,
    # limit_kd=1e1,
)

render_model = render_model_builder.finalize("cuda")
render_model.ground = False
render_model.gravity.assign(WP_GRAVITY_ZERO)

renderer = NewtonBackend(kinematic_mode=True)
renderer.set_model(render_model, render_model_builder)

# ------------------------------------------------------------------ #
#  Statistics buffer                                                 #
# ------------------------------------------------------------------ #
stats = {
    "loss": np.zeros(NUM_ITERS),
    "hand_q": np.zeros((NUM_ITERS, NUM_BATCH, len(model.joint_q))),
}

output_dir = Path(
    to_absolute_path(
        os.path.join(base_output_dir, f"{hand_name}_{obj_set}", obj_name)
    )
)
output_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
# Initial hand pose sampling --------------------------------------- #
# ------------------------------------------------------------------ #
def sample_init_hand_pose():
    joint_q_init = np.zeros((NUM_BATCH, NUM_JOINTS))
    for batch_idx in range(NUM_BATCH):
        points, faces = tri.sample.sample_surface(obj_mesh_tri, 1000)
        for i_s in range(1000):
            pos = np.asarray(points[i_s])
            normal = np.asarray(obj_mesh_tri.face_normals[faces[i_s]])

            rot_align = rotation_matrix_between(np.array([-1.0, 0.0, 0.0]), normal)
            if np.isnan(rot_align).any():
                continue
            rot_align = Rotation.from_matrix(rot_align)

            rand_roll_q = Quaternion._from_axis_angle(
                np.array([-1.0, 0.0, 0.0]), np.random.rand() * 2 * np.pi
            ).elements[[1, 2, 3, 0]]
            rand_roll = Rotation.from_quat(rand_roll_q)

            quat = (rot_align * rand_roll).as_quat()

            if np.linalg.norm(normal) == 0.0:
                continue
            pos = pos + 0.15 * (normal / np.linalg.norm(normal))

            joint_q_init[batch_idx, :3] = pos
            joint_q_init[batch_idx, 3:7] = quat
            joint_q_init[batch_idx, 7:11] = np.array(
                [6.9031e-01, 5.2765e-01, 5.7043e-01, 7.2461e-01]
            )

    # Assign to Warp arrays
    model.joint_q.assign(joint_q_init.flatten())
    state.joint_q.assign(joint_q_init.flatten())


# ------------------------------------------------------------------ #
# Grasp Optimization step                                            #
# ------------------------------------------------------------------ #
def optimize(it):
    if it >= NUM_ITERS:
        return
    # Iterative optimisation --------------------------------------- #
    tape.zero()

    cutoff = 8000
    pad_val = ((cutoff - it) / cutoff) * 0.06
    if it > cutoff:
        pad_val = 0.0
    pad_val = max(pad_val, 0.0)
    padding.fill_(pad_val)

    # wp.synchronize()
    wp.capture_launch(grasp_optimizing_graph)

    # Update stats
    loss_np = loss.numpy()
    if np.any(np.isnan(loss_np)):
        print(f"Loss is NaN at iteration {it}")
        return

    step_it_total_loss = loss_np[0]
    stats["loss"][it] = step_it_total_loss
    stats["hand_q"][it, :, :] = model.joint_q.numpy()

    if it % 1000 == 0:
        grad_norm = np.linalg.norm(tape.gradients[model.joint_q].numpy())

        # Nicely formatted, labelled output for easier inspection
        print(
            f"[Iter {it:5d}] losses"
            f"- Total: {step_it_total_loss:.4e} | "
            f"- Base: {loss_base.numpy()[0]:.4e} | "
            f"- Hand-Obj: {loss_hand_obj_interp.numpy()[0]:.4e} | "
            f"- Pose-L2: {loss_hand_pose_l2_reg.numpy()[0]:.4e} | "
            f"- Lower: {loss_hand_pose_lower.numpy()[0]:.4e} | "
            f"- Upper: {loss_hand_pose_upper.numpy()[0]:.4e} | "
            f"- GradNorm: {grad_norm:.4e}"
        )

    # Optimize grad step
    wp.launch(
        kernel=wp_kernel_apply_grad_step,
        dim=NUM_BATCH * (NUM_JOINTS - 5),
        inputs=[
            tape.gradients[model.joint_q],
            lr,
            # (NUM_JOINTS - 5) - (it % (NUM_JOINTS - 5)) - 1,
            NUM_JOINTS,
        ],
        outputs=[model.joint_q],
        device="cuda",
    )


def render_batches(it, save_results=False):
    # ------------------------------------------------------------------ #
    #  Render and save successful optimisation results                            #
    # ------------------------------------------------------------------ #
    if (np.any(np.isnan(stats["loss"])) or np.any(np.isinf(stats["loss"]))):
        return
    if save_results:
        model_joint_q_np = model.joint_q.numpy().reshape(NUM_BATCH, -1)

    for batch_idx in range(NUM_BATCH):
        # Save joint configuration
        if save_results:
            uid = uuid.uuid4().hex
            path_npy = output_dir / f"{uid}.npy"
            np.save(path_npy, model_joint_q_np[batch_idx, :])

        # ------------------------------------------------------ #
        #  Final grasp render                                    #
        # ------------------------------------------------------ #
        if RENDER_FINAL:
            render_batch(batch_idx, model.joint_q)

        # ------------------------------------------------------ #
        #  Optimisation trajectory render                         #
        # ------------------------------------------------------ #
        elif RENDER_TRAJ:
            if it % 50 != 0:
                continue
            render_batch(batch_idx, wp.array(stats["hand_q"][it, batch_idx, :], dtype=wp.float32))


def render_batch(batch_idx: int, batch_joint_q: wp.array(dtype=float)) -> None:
    # Update [renderer.joint_target_controls]
    with wp.ScopedDevice(renderer.wp_device):
        # Hand joints
        wp.launch(
            kernel=wp_kernel_copy_joint_q_for_render,
            dim=NUM_JOINTS,
            inputs=[batch_joint_q, batch_idx, NUM_JOINTS],
            outputs=[renderer.joint_target_qs],
            device="cuda",
        )

        # Save hand base pose from [state.body_q]
        if False:
            hand_base_q_start = batch_idx * render_model_builder.body_count
            hand_base_transform = state.body_q[hand_base_q_start:hand_base_q_start + 1].numpy().squeeze()
            hand_base_pose = wp.transform(wp.vec3(hand_base_transform[:3]), wp.quat(hand_base_transform[3:]))

        # Step the renderer using [renderer.joint_target_qs]
        renderer.step()
        renderer.render(obj_meshes)

        # Slow down a bit for leisure views
        renderer.sleep(0.25)


# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    for b in range(NUM_BATCH_PER_OBJ):
        print(b, "Sampling initial hand poses...")
        sample_init_hand_pose()

        print(b, "Start grasp optimization...")
        for _ in range(NUM_ITERS):
            optimize(_)

        print(b, "Visualizing Grasps...")
        for _ in range(NUM_ITERS):
            with wp.ScopedTimer("render", active=False):
                render_batches(_, save_results=True)

    # Close viewer
    renderer.viewer.close()
