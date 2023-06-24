import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from TradingEnvironment import TradingEnvironment

models_dir = "models/PPO/Delivery"
logdir = "logs/Delivery"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

portfolioValue = 500000
tradeTypeDelivery = True
deliveryPeriod = 14


def trainModelOnASingleDataset():
    TIMESTAMPS = 10000

    # Create the trading environment
    env = TradingEnvironment(
        historicalDataPath=r"D:\PycharmProjects\Paper Trading -- Reinforcement Learning\Dataset\NIFTY50_all.csv",
        portfolioValue=portfolioValue,
        tradeTypeDelivery=tradeTypeDelivery,
        deliveryPeriod=deliveryPeriod)

    # Wrap the environment with DummyVecEnv for compatibility with Stable Baselines3
    # env = DummyVecEnv([lambda: env])
    env.reset()

    # Create the PPO model
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, learning_rate=0.002, ent_coef=0.5, n_epochs=32,
                batch_size=16, gamma=0.99, clip_range=0.01, policy_kwargs={'net_arch': [32, 64, 128]})

    for i in range(1, 100):
        model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/{TIMESTAMPS * i}")


def trainModelWithAllDatasetsAvailable():
    """ Training on multiple datasets """

    TIMESTAMPS = 10000

    allDataset = os.listdir(r"D:\PycharmProjects\Paper Trading -- Reinforcement Learning\Dataset\\")

    for i, dataset in zip(range(1, len(allDataset)), allDataset):
    # for i, dataset in zip(range(1, 4), allDataset):
        # Create the trading environment
        env = TradingEnvironment(
            historicalDataPath=r"D:\PycharmProjects\Paper Trading -- Reinforcement Learning\Dataset\\" + dataset,
            portfolioValue=portfolioValue,
            tradeTypeDelivery=tradeTypeDelivery,
            deliveryPeriod=deliveryPeriod)

        env.reset()

        # Create the PPO model
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, learning_rate=0.0001, ent_coef=0.5,
                    n_epochs=32,
                    batch_size=64, gamma=0.99, clip_range=0.01, policy_kwargs={'net_arch': [64, 64, 128]})

        # Wrap the environment with DummyVecEnv for compatibility with Stable Baselines3
        env = DummyVecEnv([lambda: env])

        model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/{TIMESTAMPS * i}")


trainModelOnASingleDataset()
