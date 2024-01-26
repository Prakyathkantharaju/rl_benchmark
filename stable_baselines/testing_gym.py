
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
import time, multiprocessing
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000000,
    "env_name": None
}

def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)  # record stats such as returns
    return env


def train(env_id):
    start_time = time.perf_counter()
    config["env_name"] = env_id
    run = wandb.init(
        name=f"{config['env_name']}-ppo-sb3",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    env = DummyVecEnv([make_env] * 10)
    model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    run.finish()
    end_time = time.perf_counter()
    return end_time - start_time



def worker(env):
    return train(env)

def main(cfg):  # noqa: F821
    env_list = ["HalfCheeta-v4", 
                "Hopper-v4", 
                "HumanoidStandup-v2", 
                "Ant-v4", 
                "Humanoid-v4"]
    times = [worker(env) for env in env_list]
    return {env: time_ for env, time_ in zip(env_list, times)}

if __name__ == "__main__":
    results = main()

    import json
    with open("results.json", "w") as f:
        json.dump(results, f)
