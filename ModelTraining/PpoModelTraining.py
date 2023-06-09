import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from PaperTradingEnv.TradingEnvironment import TradingEnvironment

models_dir = "models/PPO/"
best_model_dir = "models/PPO/BestModel"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(best_model_dir):
    os.makedirs(best_model_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

portfolioValue = 1000000
tradeTypeDelivery = False
deliveryPeriod = 14


def trainModelOnASingleDataset():
    TIMESTAMPS = 1000

    # Create the trading environment
    env = TradingEnvironment(
        historicalDataPath=r"D:\PycharmProjects\Paper Trading -- Reinforcement Learning\Dataset\Nifty\BAJAJFINSV.csv",
        portfolioValue=portfolioValue,
        tradeTypeDelivery=tradeTypeDelivery,
        deliveryPeriod=deliveryPeriod)

    env.reset()

    eval_callback = EvalCallback(env,
                                 best_model_save_path=best_model_dir,  # Directory to save the best model
                                 log_path=logdir,  # Directory to save evaluation logs
                                 eval_freq=TIMESTAMPS,  # Evaluate the agent every `TIMESTAMPS` timesteps
                                 n_eval_episodes=5,  # Number of episodes to evaluate the agent
                                 deterministic=True,  # Use deterministic actions for evaluation
                                 render=False)  # Do not render the environment during evaluation

    # Wrap the environment with DummyVecEnv for compatibility with Stable Baselines3
    env = DummyVecEnv([lambda: env])

    # Create the PPO model
    # model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, learning_rate=0.5, ent_coef=0.5, n_epochs=64,
    #             batch_size=32, gamma=0.99, clip_range=0.001, policy_kwargs={'net_arch': [32, 64, 64, 128]})
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

    for i in range(1, 101):
        model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, tb_log_name="PPO", callback=eval_callback)
        model.save(f"{models_dir}/{TIMESTAMPS * i}")
        print(f"Done: {i}")


def trainModelWithAllDatasetsAvailable():
    """ Training on multiple datasets """

    TIMESTAMPS = 10000

    allDataset = os.listdir(r"D:\PycharmProjects\Paper Trading -- Reinforcement Learning\Dataset\Nifty\\")

    for i, dataset in zip(range(1, len(allDataset) + 1), allDataset):
        # Create the trading environment
        env = TradingEnvironment(
            historicalDataPath=r"D:\PycharmProjects\Paper Trading -- Reinforcement Learning\Dataset\Nifty\\" + dataset,
            portfolioValue=portfolioValue,
            tradeTypeDelivery=tradeTypeDelivery,
            deliveryPeriod=deliveryPeriod)

        eval_callback = EvalCallback(env,
                                     best_model_save_path=best_model_dir,  # Directory to save the best model
                                     log_path=logdir,  # Directory to save evaluation logs
                                     eval_freq=TIMESTAMPS,  # Evaluate the agent every `TIMESTAMPS` timesteps
                                     n_eval_episodes=10,  # Number of episodes to evaluate the agent
                                     deterministic=True,  # Use deterministic actions for evaluation
                                     render=False)  # Do not render the environment during evaluation

        # Wrap the environment with DummyVecEnv for compatibility with Stable Baselines3
        env = DummyVecEnv([lambda: env])
        env.reset()

        # Create the PPO model
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, learning_rate=0.5, ent_coef=0.5, n_epochs=64,
                    batch_size=32, gamma=0.99, clip_range=0.001, policy_kwargs={'net_arch': [32, 64, 64, 128]})

        # Wrap the environment with DummyVecEnv for compatibility with Stable Baselines3
        # env = DummyVecEnv([lambda: env])

        model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, tb_log_name="PPO", callback=eval_callback)
        model.save(f"{models_dir}/{TIMESTAMPS * i}")


trainModelOnASingleDataset()
