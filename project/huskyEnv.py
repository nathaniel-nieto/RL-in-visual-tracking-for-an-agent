import pybullet as p
import pybullet_data
import numpy as np
import cv2
import math
import time
import gymnasium as gym
from gymnasium import spaces


class HuskyTrackEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 240}

    def __init__(self, render_mode="human"):
        super(HuskyTrackEnv, self).__init__()
        self.render_mode = render_mode

        # Connect to PyBullet
        if render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.time_step = 1. / 240.
        p.setTimeStep(self.time_step)
        p.setGravity(0, 0, -9.81)

        # Action: turn left, right, or stop
        self.action_space = spaces.Discrete(3)
        # Observation: horizontal position of green block in image
        self.observation_space = spaces.Box(low=0, high=160, shape=(1,), dtype=np.float32)

        self.car = None
        self.block_id = None
        self.frame_count = 0

        #Cubo rojo id
        self.red_block_id = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("[RESET] Environment reset")
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        p.loadURDF("plane.urdf")

        # Spawn moving block
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2], rgbaColor=[0, 1, 0, 1])
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
        self.block_id = p.createMultiBody(baseMass=1,
                                          baseCollisionShapeIndex=collision,
                                          baseVisualShapeIndex=visual,
                                          basePosition=[2, 1, 0.2])
        
        # Crear cubo rojo distractor después del resetSimulation
        red_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2], rgbaColor=[1, 0, 0, 1])
        red_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
        self.red_block_id = p.createMultiBody(baseMass=1,
                                              baseCollisionShapeIndex=red_collision,
                                              baseVisualShapeIndex=red_visual,
                                              basePosition=[2.5, 2.5, 0.2])


        # Load Husky robot
        self.car = p.loadURDF("husky/husky.urdf", basePosition=[0, 1, 0.1])
        self.frame_count = 0

        obs = np.array([80], dtype=np.float32)
        info = {}
        return obs, info

    def get_camera_image(self):
        pos, orn = p.getBasePositionAndOrientation(self.car)
        matrix = p.getMatrixFromQuaternion(orn)
        forward = [matrix[0], matrix[3], matrix[6]]

        eye = [pos[0] + 0.5 * forward[0],
               pos[1] + 0.5 * forward[1],
               pos[2] + 0.6]
        target = [pos[0] + forward[0],
                  pos[1] + forward[1],
                  pos[2] + 0.3]
        view = p.computeViewMatrix(eye, target, [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100)
        _, _, img, _, _ = p.getCameraImage(160, 120, view, proj)
        return np.reshape(img, (120, 160, 4))[:, :, :3]

    def detect_green(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([90, 255, 255]))
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return -1
        return int(moments["m10"] / moments["m00"])

    def move_block(self):
        # Movimiento fluido y semialeatorio para el cubo verde
        x = 2.0 + 0.5 * math.sin(self.frame_count * 0.05 + math.sin(self.frame_count * 0.01))
        y = 1.0 + 0.5 * math.cos(self.frame_count * 0.05 + math.cos(self.frame_count * 0.015))
        p.resetBasePositionAndOrientation(self.block_id, [x, y, 0.2], [0, 0, 0, 1])

        # Movimiento suave en X para el cubo rojo
        rx = 2.5 + 0.8 * math.sin(self.frame_count * 0.07 + math.sin(self.frame_count * 0.01))
        ry = 2.5 + 0.5 * math.cos(self.frame_count * 0.07 + math.cos(self.frame_count * 0.015))
        p.resetBasePositionAndOrientation(self.red_block_id, [rx, ry, 0.2], [0, 0, 0, 1])

    #def move_block(self):
        #x = np.random.uniform(2, 3.5)
        #y = np.random.uniform(-0.5, 2.5)
        #p.resetBasePositionAndOrientation(self.block_id, [x, y, 0.2], [0, 0, 0, 1])


    def _set_wheel_velocities(self, vels):
        for j, v in zip([2, 3, 4, 5], vels):
            p.setJointMotorControl2(self.car, j, p.VELOCITY_CONTROL, targetVelocity=v, force=1000)

    def step(self, action):
        print(f"[STEP] Action: {action}")
        if action == 0:
            self._set_wheel_velocities([-2, 2, -2, 2])
        elif action == 1:
            self._set_wheel_velocities([2, -2, 2, -2])
        else:
            self._set_wheel_velocities([0, 0, 0, 0])

        self.move_block()
        for _ in range(10):
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(self.time_step)

        self.frame_count += 1

        image = self.get_camera_image()
        cx = self.detect_green(image)
        error = abs(cx - 80) if cx != -1 else 80
        #reward = -error / 80.0

        # Nueva función de recompensa más informativa
        if cx == -1:
            reward = -1.0  # Penaliza fuertemente si no se ve el cubo
        else:
            normalized_error = abs(cx - 80) / 80.0  # error entre 0 y 1
            reward = 1.0 - normalized_error  # recompensa máxima si el cubo está en el centro

        done = False
        if cx == -1 or error > 100 or self.frame_count > 2400:
            done = True
            print("[DONE] Termination condition met.")

        print(f"[OBS] cx: {cx}, reward: {reward:.2f}")
        return np.array([cx if cx != -1 else 80], dtype=np.float32), reward, done, False, {}


    def close(self):
        print("[CLOSE] Environment closed")
        p.disconnect()

