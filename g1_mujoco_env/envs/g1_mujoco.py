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

        # need to reorder actuators to fit mjlab's order and offsets
        mjlab_joint_order = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint',
            'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
            'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint',
            'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
        ]

        mjlab_offset = [-0.3120, 0.0000, 0.0000, 0.6690, -0.3630, 0.0000,
                        -0.3120, 0.0000, 0.0000, 0.6690, -0.3630, 0.0000,
                         0.0000, 0.0000, 0.0000, 0.2000,  0.2000, 0.0000,
                         0.6000, 0.0000, 0.0000, 0.0000,  0.2000, -0.2000,
                         0.0000, 0.6000, 0.0000, 0.0000,  0.0000]

        mjlab_scale  = [0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
                        0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
                        0.5475, 0.4386, 0.4386, 0.4386, 0.4386, 0.4386,
                        0.4386, 0.4386, 0.0745, 0.0745, 0.4386, 0.4386,
                        0.4386, 0.4386, 0.4386, 0.0745, 0.0745]

        # Build mapping from mjlab order -> raw actuator index
        self.raw_actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]

        # reorder offset and scale to match raw actuator order
        self.default_ctrl  = np.zeros(self.model.nu)
        self.action_scale  = np.ones(self.model.nu)
        self.mjlab_to_raw  = {}

        for mjlab_idx, joint_name in enumerate(mjlab_joint_order):
            raw_idx = self.raw_actuator_names.index(joint_name)
            self.mjlab_to_raw[mjlab_idx] = raw_idx
            self.default_ctrl[raw_idx] = mjlab_offset[mjlab_idx]
            self.action_scale[raw_idx] = mjlab_scale[mjlab_idx]

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

        # set base height
        self.data.qpos[2] = 0.76
        
        # set joint positions using raw actuator -> joint mapping
        for raw_idx in range(self.model.nu):
            act_name = self.raw_actuator_names[raw_idx]
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, act_name)
            if joint_id >= 0:
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = self.default_ctrl[raw_idx]

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
        # action comes in mjlab's joint order, reorder to raw actuator order
        action_raw = np.zeros(self.model.nu)
        for mjlab_idx, raw_idx in self.mjlab_to_raw.items():
            action_raw[raw_idx] = action[mjlab_idx]
        
        target_ctrl = self.default_ctrl + action_raw * self.action_scale
        target_ctrl = np.clip(target_ctrl,
                              self.model.actuator_ctrlrange[:, 0],
                              self.model.actuator_ctrlrange[:, 1])

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
