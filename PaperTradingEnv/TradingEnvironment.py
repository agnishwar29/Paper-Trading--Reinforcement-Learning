import gym
import numpy as np

from HistoricalDataHandler.HistoricalDataProcessor import HistoricalDataProcessor


class TradingEnvironment(gym.Env):

    def __init__(self, historicalDataPath, portfolioValue, tradeTypeDelivery=False, deliveryPeriod=None):

        self.__historicalDataPath = historicalDataPath
        self.__portfolioValue = portfolioValue
        self.__tradeTypeDelivery = tradeTypeDelivery
        self.__deliveryPeriod = deliveryPeriod

        self.__buyFlag = False
        self.__sellFlag = False

        self.data = self.getPreprocessedHistoricalData()

        self.current_step = 0
        self.max_steps = len(self.data)

        self.action_space = gym.spaces.Discrete(3)  # 0: Sell, 1: Buy, 2: Hold

        # Initialize the position to zero
        self.position = 0
        self.bank = 0.0
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

    def __handleBuyAction(self, current_data):
        reward = 0

        if self.portfolio_value >= current_data['Close']:
            if self.position == 0:
                self.portfolio_value -= current_data['Close'] * 5
                self.position_cost = current_data['Close'] * 5
                self.position += 1
                # reward = current_data['Open'] - current_data['Close']
                reward = 0.2
            elif self.position > 0:
                if self.position_cost > current_data['Close'] * 5:
                    # Incremental buy if market goes down and position cost is higher than current open price
                    self.portfolio_value -= current_data['Close'] * 5
                    self.position_cost += current_data['Close'] * 5
                    self.position += 1
                    # reward = current_data['Open'] - current_data['Close']
                    reward = 0.5
                elif self.position_cost <= current_data['Close'] * 5:
                    # Check if it has already bought twice in low
                    if self.position == 2:
                        reward = -1  # No reward for buying again in low
                    else:
                        self.portfolio_value -= current_data['Close'] * 5
                        self.position_cost += current_data['Close'] * 5
                        self.position += 1
                        # reward = current_data['Open'] - current_data['Close']
                        reward = 0.3

        else:
            reward = -1

        return reward

    def __handleSellAction(self, current_data):
        reward = -1
        profit = 0
        loss = 0

        sellValue = current_data['Close'] * self.position * 5

        if self.position > 0:
            sell_price = current_data['Close'] * self.position * 5
            if sell_price > self.position_cost:
                profit = sell_price - self.position_cost
                reward = 1
                # Add half of the profit to the portfolio and store the rest in the bank
                # self.portfolio_value += 0.5 * profit
                # self.bank += 0.5 * profit
                self.portfolio_value += sell_price

            else:
                loss = self.position_cost - sell_price
                self.portfolio_value += sell_price
                reward = -1  # -5 reward for selling at a loss

            self.position_cost = 0

        self.position = 0

        return reward, profit, sellValue, loss

    def step(self, action):
        self.current_step += 1
        done = False
        reward = 0
        loss = 0
        profit = 0

        # Get the current market data
        current_data = self.data.iloc[self.current_step]

        if self.current_step + 1 == len(self.data) or self.portfolio_value < current_data['Close']:
            done = True

        # TODO: remove
        sellValue = 0
        # Take action based on the agent's decision
        if action == 0:  # Sell action
            reward, profit, sellValue, loss = self.__handleSellAction(current_data=current_data)
            # print(f"Sell action || Positions: {self.position}")

        elif action == 1:  # Buy action
            reward = self.__handleBuyAction(current_data=current_data)
            # print(f"Buy action || Positions: {self.position}")

        elif action == 3:  # Hold
            reward = 0.5

        info = {'portfolio_value': self.portfolio_value, 'profit': profit, 'position': self.position, 'bank': self.bank,
                'position_cost': self.position_cost, 'sellValue': sellValue, 'loss': loss}

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        # Retrieve the current market data for the observation
        observation = self.data.iloc[self.current_step].values
        mean = np.mean(self.data.values, axis=0)
        std = np.std(self.data.values, axis=0)
        normalized_observation = (observation - mean) / std

        normalized_observation = normalized_observation.astype(np.float32)
        return normalized_observation
