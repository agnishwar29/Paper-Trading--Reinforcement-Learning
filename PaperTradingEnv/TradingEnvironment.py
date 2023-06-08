
import gym
import numpy as np


from HistoricalDataHandler.HistoricalDataProcessor import HistoricalDataProcessor


class TradingEnvironment(gym.Env):

    def __init__(self, historicalDataPath):
        self.__historicalDataPath = historicalDataPath
        self.data = self.getPreprocessedHistoricalData()

        self.current_step = 0
        self.max_steps = len(self.__historicalDataPath)

        self.action_space = gym.spaces.Discrete(2)  # 0: Sell, 1: Buy
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=float)

    def getPreprocessedHistoricalData(self):

        dataProcessor = HistoricalDataProcessor()
        dataProcessor.specifyHistoricalDataPath(path=self.__historicalDataPath)

        return dataProcessor.getProcessedData()

    # def reset(self):
    #     self.current_step = 0
    #     return self._get_observation()
    #
    # def step(self, action):
    #     self.current_step += 1
    #     done = False
    #     reward = 0
    #
    #     # Get the current market data
    #     current_data = self.data.iloc[self.current_step]
    #
    #     if self.current_step + 1 == len(self.data):
    #         done = True
    #
    #         # Take action based on the agent's decision
    #         if action == 0:  # Sell action
    #             # Calculate the reward based on the sell action
    #             reward = current_data['Close'] - current_data['Open']
    #         elif action == 1:  # Buy action
    #             # Calculate the reward based on the buy action
    #             reward = current_data['Open'] - current_data['Close']
    #         else:
    #             raise ValueError("Invalid action.")
    #
    #     info = {}  # Additional information for debugging or analysis
    #
    #     return self._get_observation(), reward, done, info
    #
    # def _get_observation(self):
    #     # Retrieve the current market data for the observation
    #     observation = self.data.iloc[self.current_step].values
    #     observation = observation.astype(np.float32)
    #     return observation

    def reset(self):
        self.current_step = 0
        self.portfolio_value = 100000  # Reset the portfolio value
        self.holding_stock = False  # Reset the stock holding status
        return self._get_observation()

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Get the current market data
        current_data = self.data.iloc[self.current_step]

        reward = 0

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

        info = {'portfolio_value': self.portfolio_value}  # Additional information for debugging or analysis

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        # Retrieve the current market data for the observation
        observation = self.data.iloc[self.current_step].values
        observation = observation.astype(np.float32)
        return observation

