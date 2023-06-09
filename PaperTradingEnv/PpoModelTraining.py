import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from TradingEnvironment import TradingEnvironment

models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create the trading environment
env = TradingEnvironment(
    historicalDataPath=r"D:\PycharmProjects\Paper Trading -- Reinforcement Learning\Dataset\ITC.csv", portfolioValue=10000)

# Wrap the environment with DummyVecEnv for compatibility with Stable Baselines3
env = DummyVecEnv([lambda: env])

TIMESTAMPS = 1000

# Create the PPO model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, learning_rate=0.0001, ent_coef=0.5, n_epochs=32,
            batch_size=64, gamma=0.99, clip_range=0.01, policy_kwargs={'net_arch': [32, 64, 128]})


for i in range(1, 5):
    model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTAMPS * i}")

""" Training on multiple datasets """
# for i, dataset in zip(range(1, 53), os.listdir(r"D:\PycharmProjects\Paper Trading -- Reinforcement Learning\Dataset\\")):
#     # Create the trading environment
#     env = TradingEnvironment(historicalDataPath=r"D:\PycharmProjects\Paper Trading -- Reinforcement Learning\Dataset\\" + dataset, portfolioValue=10000)
#
#     env.reset()
#
#     historicalData = env.getPreprocessedHistoricalData()
#
#     # Create the PPO model
#     model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
#
#     # Wrap the environment with DummyVecEnv for compatibility with Stable Baselines3
#     env = DummyVecEnv([lambda: env])
#
#     model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, tb_log_name="PPO")
#     model.save(f"{models_dir}/{TIMESTAMPS * i}")



