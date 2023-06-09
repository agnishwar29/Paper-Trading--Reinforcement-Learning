
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

        # Initialize the position to zero
        self.position = 0

        self.position_cost = 0.0  # Initialize position cost to 0

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
            if self.position > 0:
                sell_price = current_data['Close'] * self.position
                if sell_price > self.position_cost:
                    self.portfolio_value += sell_price
                    self.position = 0
                    reward = current_data['Close'] - current_data['Open']
                else:
                    reward = 0  # No reward for selling at a loss
        elif action == 1:  # Buy action
            if self.position == 0:
                self.portfolio_value -= current_data['Close']
                self.position_cost = current_data['Close']
                self.position += 1
                reward = current_data['Open'] - current_data['Close']
            elif self.position > 0 and current_data['Close'] < current_data['Open']:
                # Incremental buy if market goes down
                self.portfolio_value -= current_data['Close']
                self.position_cost += current_data['Close']
                self.position += 1
                reward = current_data['Open'] - current_data['Close']

        info = {'portfolio_value': self.portfolio_value, 'position': self.position}

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        # Retrieve the current market data for the observation
        observation = self.data.iloc[self.current_step].values
        mean = np.mean(self.data.values, axis=0)
        std = np.std(self.data.values, axis=0)
        normalized_observation = (observation - mean) / std

        normalized_observation = normalized_observation.astype(np.float32)
        return normalized_observation

