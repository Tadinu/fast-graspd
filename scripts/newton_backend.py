import time
from typing import Optional, Callable
import numpy as np
import warp as wp
import newton


@wp.kernel
def wp_kernel_default_set_joint_targets(
        controls: wp.array(dtype=wp.float32, ndim=1),
        joint_ids: wp.array(dtype=wp.int32, ndim=1),
        joint_qd_starts: wp.array(dtype=wp.int32),
        joint_limit_lowers: wp.array(dtype=wp.float32),
        joint_limit_uppers: wp.array(dtype=wp.float32),
        sim_dt: float,
        kinematic_mode: bool,
        # outputs
        joint_targets: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    joint_id = joint_ids[tid]

    # Move joints
    if kinematic_mode:
        joint_targets[joint_id] = controls[joint_id]
    else:
        joint_dof_id = joint_qd_starts[joint_id]
        joint_targets[joint_dof_id] = wp.clamp(controls[joint_dof_id],
                                               joint_limit_lowers[joint_dof_id], joint_limit_uppers[joint_dof_id])
    # wp.printf("%d %d %f\n", joint_id, joint_dof_id, joint_targets[joint_dof_id])


class NewtonBackend:
    def __init__(self, num_substeps: int = 2,
                 is_batched: bool = False,
                 kinematic_mode: bool = False,
                 headless: bool = False,
                 kernel_set_joint_targets: Optional[Callable] = None):
        # Settings
        self.fps: int = 50
        self.frame_dt: float = 1.0 / self.fps

        self.sim_time: float = 0.0
        if not kinematic_mode:
            assert num_substeps > 1, f"Newton backend in physics mode requires substeps as larger than 1 for the warmup!"
        self.sim_substeps: int = num_substeps
        self.sim_dt: float = self.frame_dt / self.sim_substeps
        self.wp_device = wp.get_device()

        # Rollout
        self.is_batched = is_batched
        self.wp_is_batched = wp.ones(1, dtype=int) if self.is_batched else wp.zeros(1, dtype=int)

        # Model
        self.model: newton.Model = None
        self.model_builder: newton.ModelBuilder = None
        self.kinematic_mode: bool = kinematic_mode

        # Robot
        self.joint_names: list[str] = []
        self.joint_ids: wp.array(dtype=wp.int32) = None
        self.joint_target_controls: wp.array(dtype=wp.float32) = None
        self.kernel_set_joint_targets = kernel_set_joint_targets if kernel_set_joint_targets \
            else wp_kernel_default_set_joint_targets

        # Solver
        self.solver = None

        # States
        self.state_0: newton.State = None
        self.state_1: newton.State = None
        self.model_ctrl: newton.Control = None
        self.contacts: newton.Contacts = None

        # Viewer
        self.headless = headless
        self.viewer = None

        # Capture
        self.graph = None

    def set_model(self, model: newton.Model, model_builder: newton.ModelBuilder,
                  joint_names: Optional[list[str]] = None) -> None:
        # Model evaluation
        self.model = model
        self.model_builder = model_builder

        # Joint ids
        self.joint_names = joint_names if joint_names else model_builder.joint_key
        self.joint_ids = wp.array(np.arange(len(model_builder.joint_q)), dtype=wp.int32) if self.kinematic_mode \
            else wp.array([model_builder.joint_key.index(jname) for jname in self.joint_names], dtype=wp.int32)

        # Joint target controls
        self.joint_target_qs = wp.zeros(len(model_builder.joint_q), dtype=wp.float32)
        self.joint_target_controls = wp.zeros(model_builder.joint_dof_count, dtype=wp.float32)

        # Eval model fk
        self.state_0 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_0)

        # Model solver
        self.solver = newton.solvers.SolverMuJoCo(model,
                                                  solver="newton",
                                                  integrator="implicitfast",
                                                  njmax=200,
                                                  nconmax=150,
                                                  impratio=10.0,
                                                  cone="elliptic",
                                                  iterations=100,
                                                  ls_iterations=50,
                                                  use_mujoco_cpu=False)

        # States
        self.state_1 = model.state()
        self.model_ctrl = model.control()
        self.contacts = model.collide(self.state_0)

        # Viewer
        # NOTE: Thought ViewerGL has a `headless` param, but it seems it still causes trouble to the headful one.
        # -> Disable for now. TODO: Find out why!
        self.viewer = newton.viewer.ViewerGL() if not self.headless else None
        if self.viewer:
            self.viewer.set_model(model)

        # Capture (always the last, only once the model is fully setup)
        self.graph = None
        self.capture()

    def capture(self) -> None:
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                if not self.is_batched:
                    self._prepare_model_ctrl()
                self._step_simulation()
            self.graph = capture.graph

    def _prepare_model_ctrl(self) -> None:
        with wp.ScopedDevice(self.wp_device):
            wp.launch(
                self.kernel_set_joint_targets,
                dim=len(self.model_builder.joint_q) if self.kinematic_mode else self.model_builder.joint_dof_count,
                inputs=[
                    self.joint_target_qs if self.kinematic_mode else self.joint_target_controls,
                    self.joint_ids,
                    self.model.joint_qd_start,
                    self.model.joint_limit_lower,
                    self.model.joint_limit_upper,
                    self.sim_dt,
                    self.kinematic_mode
                ],
                outputs=[self.model.joint_q if self.kinematic_mode else self.model_ctrl.joint_target_pos],
            )

    def _step_simulation(self) -> None:
        if self.kinematic_mode:
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        else:
            self.contacts = self.model.collide(self.state_0)
            for i in range(self.sim_substeps):
                self.state_0.clear_forces()
                # apply forces to the model for picking, wind, etc
                # NOTE: [self.viewer] does not change at run time, so no need for [wp.capture_if] here
                if self.viewer:
                    self.viewer.apply_forces(self.state_0)

                # update the solver since we have updated the joint parent transforms
                self.solver.notify_model_changed(newton.solvers.SolverNotifyFlags.JOINT_PROPERTIES)

                # Contacts
                self.solver.step(self.state_0, self.state_1, self.model_ctrl, self.contacts, self.sim_dt)

                # swap states
                self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Step the backend"""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            if not self.is_batched:
                self._prepare_model_ctrl()
            self._step_simulation()

        self.sim_time += self.frame_dt

    def reset(self):
        self.solver.step(self.state_0, self.state_1, self.model_ctrl, self.contacts, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self, obj_meshes: Optional[dict[str, wp.Mesh]] = None):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        if obj_meshes:
            for obj_name, obj_mesh in obj_meshes.items():
                self.viewer.log_mesh(
                    name=obj_name,
                    points=obj_mesh.points,
                    indices=obj_mesh.indices,
                )
        self.viewer.end_frame()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def spin(self) -> None:
        while self.viewer.is_running():
            if not self.viewer.is_paused():
                with wp.ScopedTimer("step", active=False):
                    self.step()

                with wp.ScopedTimer("render", active=False):
                    self.render()

        # Close viewer
        self.viewer.close()
