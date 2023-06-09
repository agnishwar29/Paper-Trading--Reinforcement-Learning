import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from TradingEnvironment import TradingEnvironment

# Create the trading environment
env = TradingEnvironment(
    historicalDataPath=r"D:\PycharmProjects\Paper Trading -- Reinforcement Learning\Dataset\ITC.csv", portfolioValue=100000)

historicalData = env.getPreprocessedHistoricalData()
num_steps = len(historicalData)

# Wrap the environment with DummyVecEnv for compatibility with Stable Baselines3
env = DummyVecEnv([lambda: env])
models_dir = "models/PPO"
model_path = f"{models_dir}/4000.zip"
model = PPO.load(model_path, env=env)

# Perform paper trading with real-time visualization
obs = env.reset()

fig, ax = plt.subplots()
ax.plot(historicalData['Close'])

scatter_sell = ax.scatter([], [], color='red', marker='v')
scatter_buy = ax.scatter([], [], color='green', marker='^')

plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Paper Trading')

portfolio_values = []  # List to store portfolio values
profits = []  # List to store profit values
losses = []  # List to store loss values

for step in range(num_steps):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)

    # Visualization: Update the scatter plots based on the action
    if action == 0:  # Sell action
        scatter_sell.set_offsets((step, historicalData['Close'].iloc[step]))
    elif action == 1:  # Buy action
        scatter_buy.set_offsets((step, historicalData['Close'].iloc[step]))

    # Calculate the profit in real time
    portfolio_value = info[0]['portfolio_value']
    portfolio_values.append(portfolio_value)

    if done:
        break

    # Calculate profit/loss values
    if len(portfolio_values) > 1:
        profit = portfolio_values[-1] - portfolio_values[-2]
        if profit >= 0:
            profits.append(profit)
            losses.append(0)
        else:
            profits.append(0)
            losses.append(profit)

    # Set the label for the scatter plot
    label_text = f"Portfolio Value: {portfolio_value:.2f}\nProfit: {sum(profits):.2f}\nLoss: {sum(losses):.2f}"
    scatter_sell.set_label(label_text)

    # Call the legend() function to display the label
    ax.legend()

    # Update the plot
    # plt.pause(0.5)  # Adjust the pause duration to control the speed of trading

# Plot the portfolio value, profit, and loss over time
plt.figure(figsize=(10, 6))  # Set the figure size
plt.plot(portfolio_values, label='Portfolio Value')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Portfolio Value, Profit, and Loss over Time')
plt.legend()
plt.tight_layout()  # Adjust the spacing of the plot elements
plt.show()

print('Final Portfolio Value:', portfolio_values[-1])
print('Total Profit:', sum(profits))
print('Total Loss:', sum(losses))