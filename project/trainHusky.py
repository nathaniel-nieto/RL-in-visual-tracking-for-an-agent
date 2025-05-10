from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from huskyEnv import HuskyTrackEnv

# Custom callback to print episode rewards during training
class PrintRewardCallback(BaseCallback):
    def __init__(self, check_freq=1, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if len(self.locals["infos"]) > 0:
            info = self.locals["infos"][0]
            if "episode" in info:
                reward = info["episode"]["r"]
                print(f"Step {self.num_timesteps}: Episode reward = {reward:.2f}")
        return True

def main():
    # Create environment (render GUI for debugging)
    env = HuskyTrackEnv(render_mode="none")
 
    # Optional: check that environment follows Gym API
    check_env(env, warn=True)

    # Initialize model (MLP policy)
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")


    total_timesteps = 2400 * 10  # Around 30 episodes (assuming 2400 steps/episode)
    model.learn(total_timesteps=total_timesteps, callback=PrintRewardCallback())

    # Save the model
    model.save("ppo_husky_track1")

    # Close environment
    env.close()

if __name__ == "__main__":
    main()
