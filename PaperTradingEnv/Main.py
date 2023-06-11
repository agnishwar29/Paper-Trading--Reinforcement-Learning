import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from TradingEnvironment import TradingEnvironment


datasetFilepath = r"D:\PycharmProjects\Paper Trading -- Reinforcement Learning\Dataset\\"

datasets = ['BAJAJ-AUTO.csv', 'BAJAJFINSV.csv', 'TATASTEEL.csv', 'HDFC.csv', 'HDFCBANK.csv', 'NESTLEIND.csv', 'TCS.csv']


# Create the trading environment
env = TradingEnvironment(
    historicalDataPath=r"D:\PycharmProjects\Paper Trading -- Reinforcement Learning\Dataset\BAJAJFINSV.csv",
    portfolioValue=100000, tradeTypeDelivery=True, deliveryPeriod=14)

historicalData = env.getPreprocessedHistoricalData()
num_steps = len(historicalData)

# Wrap the environment with DummyVecEnv for compatibility with Stable Baselines3
env = DummyVecEnv([lambda: env])
models_dir = "models/PPO"
model_path = f"{models_dir}/1620000.zip"
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

bank_balance = 0.0  # Initialize bank balance

for step in range(num_steps):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)

    # Update the bank balance
    bank_balance = info[0]['bank']

    # Visualization: Update the scatter plots based on the action
    if action == 0:  # Sell action
        scatter_sell.set_offsets((step, historicalData['Close'].iloc[step]))
    elif action == 1:  # Buy action
        scatter_buy.set_offsets((step, historicalData['Close'].iloc[step]))

    # Calculate the profit in real time
    portfolio_value = info[0]['portfolio_value']
    portfolio_values.append(portfolio_value)

    profits.append(info[0]['profit'])

    print("\nAction: ", "buy" if action == 1 else "sell")
    print(f"Position: {info[0]['position']} || Position cost: {info[0]['position_cost']} || Portfolio value: {info[0]['portfolio_value']}")

    if action == 0:
        print(f"Sell value: {info[0]['sellValue']} || Portfolio value: {info[0]['portfolio_value']}")
        if info[0]['loss'] != 0:
            print(f"Loss: {info[0]['loss']}")
        losses.append(info[0]['loss'])

    if done:
        break

    # Set the label for the scatter plot
    label_text = f"Portfolio Value: {portfolio_value:.2f}\nProfit: {sum(profits):.2f}"
    scatter_sell.set_label(label_text)

    # Call the legend() function to display the label
    ax.legend()

    # Update the plot
    plt.pause(0.001)  # Adjust the pause duration to control the speed of trading


# Calculate the final portfolio value including the bank balance
final_portfolio_value = portfolio_values[-1] + bank_balance

"""
for dataset in datasets:

    currData = datasetFilepath + dataset

    # Create the trading environment
    env = TradingEnvironment(
        historicalDataPath=currData,
        portfolioValue=100000, tradeTypeDelivery=True, deliveryPeriod=14)

    historicalData = env.getPreprocessedHistoricalData()
    num_steps = len(historicalData)

    # Wrap the environment with DummyVecEnv for compatibility with Stable Baselines3
    env = DummyVecEnv([lambda: env])
    models_dir = "models/PPO"
    model_path = f"{models_dir}/1450000.zip"
    model = PPO.load(model_path, env=env)

    # Perform paper trading with real-time visualization
    obs = env.reset()

    fig, ax = plt.subplots()
    ax.plot(historicalData['Close'])

    scatter_sell = ax.scatter([], [], color='red', marker='v')
    scatter_buy = ax.scatter([], [], color='green', marker='^')

    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f'Paper Trading {dataset[:-4]}')

    portfolio_values = []  # List to store portfolio values
    profits = []  # List to store profit values
    losses = []  # List to store loss values

    bank_balance = 0.0  # Initialize bank balance

    for step in range(num_steps):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        # Update the bank balance
        bank_balance = info[0]['bank']

        # Visualization: Update the scatter plots based on the action
        if action == 0:  # Sell action
            scatter_sell.set_offsets((step, historicalData['Close'].iloc[step]))
        elif action == 1:  # Buy action
            scatter_buy.set_offsets((step, historicalData['Close'].iloc[step]))

        # Calculate the profit in real time
        portfolio_value = info[0]['portfolio_value']
        portfolio_values.append(portfolio_value)

        profits.append(info[0]['profit'])

        print("\nAction: ", "buy" if action == 1 else "sell")
        print(f"Position: {info[0]['position']} || Position cost: {info[0]['position_cost']} || Portfolio value: {info[0]['portfolio_value']}")

        if action == 0:
            print(f"Sell value: {info[0]['sellValue']} || Portfolio value: {info[0]['portfolio_value']}")
            if info[0]['loss'] != 0:
                print(f"Loss: {info[0]['loss']}")
            losses.append(info[0]['loss'])

        if done:
            break

        # Set the label for the scatter plot
        label_text = f"Portfolio Value: {portfolio_value:.2f}\nProfit: {sum(profits):.2f}"
        scatter_sell.set_label(label_text)

        # Call the legend() function to display the label
        ax.legend()

        # Update the plot
        plt.pause(0.01)  # Adjust the pause duration to control the speed of trading


    # Calculate the final portfolio value including the bank balance
    final_portfolio_value = portfolio_values[-1] + bank_balance

"""
