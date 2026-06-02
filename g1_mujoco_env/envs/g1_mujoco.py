from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from mjlab.entity.entity import Entity
import mjlab.asset_zoo.robots.unitree_g1.g1_constants as g1
import struct
import time


class G1MujocoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, decimation=4):
        self.num_obs = 70
        self.num_acts = 29
        self.decimation = decimation

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.num_obs,), dtype=float)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.num_acts,), dtype=float)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.robot = Entity(g1.get_g1_robot_cfg())

        # add ground
        self.ground = self.robot.spec.worldbody.add_geom()
        self.ground.type = mujoco.mjtGeom.mjGEOM_PLANE
        self.ground.size = [0, 0, 1.0]
        self.ground.pos = [0, 0, 0]
        self.ground.name = "ground"

        self.model = self.robot.spec.compile()

        # match mjlab's SimulationCfg
        self.model.opt.timestep = 0.005
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.impratio = 1.0
        self.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.model.opt.iterations = 10
        self.model.opt.tolerance = 1e-8
        self.model.opt.ls_iterations = 20
        self.model.opt.ls_tolerance = 0.01
        self.model.opt.ccd_iterations = 50

        self.data = mujoco.MjData(self.model)

        self.actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]

        self.default_ctrl = np.zeros(self.model.nu)

        for i, name in enumerate(self.actuator_names):
            joint_name = name.replace("actuator_", "")
            for pattern, pos in g1.KNEES_BENT_KEYFRAME.joint_pos.items():
                import re
                if re.fullmatch(pattern, joint_name):
                    self.default_ctrl[i] = pos

        self.action_scale = np.array([
            next(
                (v for k, v in g1.G1_ACTION_SCALE.items() if re.fullmatch(k, name.replace("actuator_", ""))),
                1.0
            )
            for name in self.actuator_names
        ])

        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

        self.target_vel = np.array([0.5, 0.0, 0.0])

    def _get_imu(self):
        pelvis_id = 1

        # data.xquat is [w, x, y, z]
        quat = self.data.xquat[pelvis_id]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        gravity_world = np.array([0.0, 0.0, -1.0])
        projected_gravity = rot.T @ gravity_world  # world -> body frame

        # angular velocity — sensor type 3 (gyro), sensordata[0:3]
        ang_vel = self.data.sensordata[0:3]

        # linear velocity — sensor type 2, sensordata[3:6]
        lin_vel = self.data.sensordata[3:6]

        # linear acceleration — sensor type 1, sensordata[6:9]
        lin_acc = self.data.sensordata[6:9]

        # imu_upvector — sensordata[9:12] (this is essentially projected gravity too)
        #up_vector = self.data.sensordata[9:12]

        return projected_gravity, ang_vel, lin_vel, lin_acc

    def _is_fallen(self):
        too_low = self.data.qpos[2] < 0.3
        too_tilted = self._get_imu()[0][2] > -0.5

        return too_low or too_tilted

    def _vel_reward(self):
        forward_vel = self.data.qvel[0]
        vel_reward = 1.0 - np.average(np.absolute(forward_vel - self.target_vel))

        return vel_reward

    def _get_obs(self):
        projected_gravity, ang_vel, lin_vel, _ = self._get_imu()

        imu_obs = np.concatenate((lin_vel, ang_vel, projected_gravity), axis=0)

        joint_pos_obs = self.data.qpos[7:]
        joint_vel_obs = self.data.qvel[6:]

        return np.concatenate((imu_obs, joint_pos_obs, joint_vel_obs, self.target_vel), axis=0)

    def _get_info(self):
        return {
            "velocity": self.data.qvel
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # set initial pose
        self.data.ctrl[:] = self.default_ctrl

        # also set qpos to match keyframe
        # position the base
        #data.qpos[2] = 0.76  # z height from KNEES_BENT_KEYFRAME
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        target_ctrl = self.default_ctrl + action * self.action_scale

        for _ in range(self.decimation):
            self.data.ctrl[:] = target_ctrl
            mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        reward = self._vel_reward()
        terminated = False

        if self._is_fallen():
            reward = -100.0
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self):
        pass

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
