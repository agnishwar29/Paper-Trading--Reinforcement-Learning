
import gym
import numpy as np


from HistoricalDataHandler.HistoricalDataProcessor import HistoricalDataProcessor


class TradingEnvironment(gym.Env):

    def __init__(self, historicalDataPath, portfolioValue):
        self.__historicalDataPath = historicalDataPath
        self.__portfolioValue = portfolioValue
        self.data = self.getPreprocessedHistoricalData()

        self.current_step = 0
        self.max_steps = len(self.data)

        self.action_space = gym.spaces.Discrete(2)  # 0: Sell, 1: Buy

        # the shape of the observation is 5 because we have Open, High, Low, Close, Volume, RSI
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6,), dtype=float)

    def getPreprocessedHistoricalData(self):

        dataProcessor = HistoricalDataProcessor()
        dataProcessor.specifyHistoricalDataPath(path=self.__historicalDataPath)

        return dataProcessor.getProcessedData()[14:]

    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.__portfolioValue  # Reset the portfolio value
        self.holding_stock = False  # Reset the stock holding status
        return self._get_observation()

    def step(self, action):
        self.current_step += 1

        done = False
        reward = 0

        # Get the current market data
        current_data = self.data.iloc[self.current_step]

        if self.current_step + 1 == len(self.data):
            done = True

        # Take action based on the agent's decision
        if action == 0:  # Sell action
            if self.holding_stock:
                self.portfolio_value += current_data['Close']  # Add the current stock price to the portfolio value
                self.holding_stock = False  # Reset the stock holding status
                reward = current_data['Close'] - current_data['Open']  # Calculate the reward based on the sell action
        elif action == 1:  # Buy action
            if not self.holding_stock:
                self.portfolio_value -= current_data['Close']  # Deduct the current stock price from the portfolio value
                self.holding_stock = True  # Set the stock holding status
                reward = current_data['Open'] - current_data['Close']  # Calculate the reward based on the buy action

        info = {'portfolio_value': self.portfolio_value}

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        # Retrieve the current market data for the observation
        observation = self.data.iloc[self.current_step].values
        mean = np.mean(self.data.values, axis=0)
        std = np.std(self.data.values, axis=0)
        normalized_observation = (observation - mean) / std

        normalized_observation = normalized_observation.astype(np.float32)
        return normalized_observation

