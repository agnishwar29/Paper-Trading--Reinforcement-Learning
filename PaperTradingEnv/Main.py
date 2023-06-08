import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from TradingEnvironment import TradingEnvironment

# Create the trading environment
env = TradingEnvironment(
    historicalDataPath=r"D:\PycharmProjects\Paper Trading -- Reinforcement Learning\Dataset\ITC.csv")

historicalData = env.getPreprocessedHistoricalData()
num_steps = len(historicalData)

# Wrap the environment with DummyVecEnv for compatibility with Stable Baselines3
env = DummyVecEnv([lambda: env])

models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Create the PPO model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTAMPS = 10000

for i in range(1, 50):
    model.learn(total_timesteps=TIMESTAMPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTAMPS * i}")

# Perform paper trading with real-time visualization
obs = env.reset()

portfolio_values = []  # List to store portfolio values over time

fig, ax = plt.subplots()
ax.plot(historicalData['Close'])

scatter_sell = ax.scatter([], [], color='red', marker='v')
scatter_buy = ax.scatter([], [], color='green', marker='^')

plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Paper Trading')

for step in range(num_steps):

    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)

    # Visualization: Update the scatter plots based on the action
    if action == 0:  # Sell action
        scatter_sell.set_offsets((step, historicalData['Close'].iloc[step]))
    elif action == 1:  # Buy action
        scatter_buy.set_offsets((step, historicalData['Close'].iloc[step]))

    # Calculate the profit in real time
    portfolio_value = info['portfolio_value']
    portfolio_values.append(portfolio_value)

    if done:
        break

    # Update the plot
    plt.pause(0.01)

# Plot the portfolio value over time
plt.figure()
plt.plot(portfolio_values)
plt.xlabel('Time')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value over Time')
plt.show()