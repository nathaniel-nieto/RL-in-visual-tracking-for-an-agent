import time
import numpy as np
from stable_baselines3 import PPO
from huskyEnv import HuskyTrackEnv

def test_model():
    # Create the environment in GUI mode
    env = HuskyTrackEnv(render_mode="human")

    # Load the trained PPO model
    model = PPO.load("ppo_husky_track1")

    obs, _ = env.reset()

    while True:
        # Use model to predict the action
        action, _ = model.predict(obs, deterministic=True)

        # Step environment with predicted action
        obs, reward, done, truncated, info = env.step(action)

        # Optional: print debug info
        print(f"Action: {action}, Reward: {reward:.2f}")

        # Reset environment if episode ends
        if done or truncated:
            obs, _ = env.reset()
            print("🔄 Episode restarted.")

        time.sleep(1. / 60.)  # slow down a bit for visibility

if __name__ == "__main__":
    test_model()
