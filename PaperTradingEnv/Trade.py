import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from TradingEnvironment import TradingEnvironment
import seaborn as sns
import multiprocessing as mp


matplotlib.use('qt5agg')


class Trade:

    def __init__(self):
        self.__datasetFilepath = r"D:\PycharmProjects\Paper Trading -- Reinforcement Learning\Dataset\Nifty\\"
        self.__stocks = ['BAJAJFINSV.csv', 'TATASTEEL.csv', 'HDFC.csv', 'HDFCBANK.csv']

        self.__models_dir = r"D:\PycharmProjects\Paper Trading -- Reinforcement Learning\ModelTraining\models\PPO"
        self.__model_path = fr"{self.__models_dir}\\66000.zip"
        self.__portfolioValue = 2000000
        self.__finalPortfolioValue = 0

    def runTrade(self):

        splitPortfolioValue = self.__portfolioValue / len(self.__stocks)

        processes = []

        for i, stock in enumerate(self.__stocks):
            process = mp.Process(target=self.trade, args=(stock, False, splitPortfolioValue))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        print("Trade done")
        print(f"Portfolio value: {self.__finalPortfolioValue}")

    def trade(self, dataset, tradeType, amount):
        env = TradingEnvironment(
            historicalDataPath=f"{self.__datasetFilepath}{dataset}",
            portfolioValue=amount, tradeTypeDelivery=tradeType, deliveryPeriod=14)

        historicalData = env.getPreprocessedHistoricalData()
        num_steps = len(historicalData)

        model = PPO.load(self.__model_path, env=env)

        obs = env.reset()

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.set()  # Apply seaborn style

        scatter_sell = ax.scatter([], [], color='red', marker='v')
        scatter_buy = ax.scatter([], [], color='green', marker='^')
        line_price, = ax.plot([], [], color='blue', label='Price')

        portfolio_values = []  # List to store portfolio values
        profits = []  # List to store profit values
        losses = []  # List to store loss values
        prices = []

        for step in range(num_steps):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)

            print(f"Stock: {dataset} || Action: {action} || Portfolio value: {info['portfolio_value']} || Position cost: {info['position_cost']}")

            # Visualization: Update the scatter plots based on the action
            if action == 0:  # Sell action
                # scatter_sell.set_offsets((step, historicalData['Close'].iloc[step]))
                scatter_sell.set_offsets((step, historicalData['Close'].iloc[step]))

            elif action == 1:  # Buy action
                # scatter_buy.set_offsets((step, historicalData['Close'].iloc[step]))
                scatter_buy.set_offsets((step, historicalData['Close'].iloc[step]))

            # Calculate the profit in real time
            portfolio_value = info['portfolio_value']
            portfolio_values.append(portfolio_value)

            price = historicalData['Close'].iloc[step]
            prices.append(price)

            profits.append(info['profit'])

            if action == 0:
                losses.append(info['loss'])

            if done:
                break

            # Set the label for the scatter plot
            label_text = f"Portfolio Value: {portfolio_value:.2f}\nHolding: {info['position'] * 15}\n" \
                         f"Profit: {sum(profits):.2f}"
            scatter_buy.set_label(label_text)

            line_price.set_data(range(step + 1), prices)

            ax.relim()  # Recalculate the data limits
            ax.autoscale_view(True, True, True)  # Autoscale the axes
            ax.legend()
            ax.set_title(f"Trade: {dataset}")
            plt.draw()
            plt.pause(0.001)

        print(f"Final portfolio value for {dataset} -> {portfolio_values[-1]} || Total profit: {sum(profits)}")


if __name__ == '__main__':
    trade = Trade()
    trade.runTrade()
