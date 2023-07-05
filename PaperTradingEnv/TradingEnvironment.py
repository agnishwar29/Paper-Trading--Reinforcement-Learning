import gym
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from HistoricalDataHandler.HistoricalDataProcessor import HistoricalDataProcessor


class TradingEnvironment(gym.Env):

    def __init__(self, historicalDataPath, portfolioValue, tradeTypeDelivery=False, deliveryPeriod=None):

        self.buyReward = 0
        self.sellReward = 0
        self.holdReward = 0
        self.__previousPositionCost = 0
        self.scaler = MinMaxScaler()
        self.__portfolio_value = None
        self.__historicalDataPath = historicalDataPath
        self.__portfolioValue = portfolioValue
        self.__tradeTypeDelivery = tradeTypeDelivery
        self.__deliveryPeriod = deliveryPeriod
        self.__deliveryPeriodCount = 0
        self.__previousActions = [0]

        # make the profit, loss as global
        self.__profit = 0
        self.__loss = 0
        self.__action = 0
        self.__reward = 0

        self.__stockChunk = 15

        self.__buyFlag = True
        self.__sellFlag = True

        self.data = self.getPreprocessedHistoricalData()

        self.current_step = 0
        self.max_steps = len(self.data)

        # Initialize the position to zero
        self.__positions = 0
        self.bank = 0.0
        self.__position_cost = 0.0  # Initialize position cost to 0

        self.action_space = gym.spaces.Discrete(3)  # 0: Sell, 1: Buy, 2: Hold

        # the shape of the observation is 5 because we have Close, Volume, RSI, EMA
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(9,), dtype=float)

    def __checkRsiSignalBuy(self, currentData):
        """
        checks if the RSI value for the current data is less than or equals to 30
        :param currentData:
        :return: True if condition satisfies else False
        """
        if currentData['RSI'] <= 30:
            return True
        return False

    def __checkRsiSignalSell(self, currentData):
        """
        checks if the RSI value for the current data is greater than or equals to 70
        :param currentData: True if condition satisfies else False
        :return:
        """
        if currentData['RSI'] >= 70:
            return True
        return False

    def __checkRsiSignalHold(self, currentData):
        """
        checks if the RSI value for the current data is greater than 30 and less than 70
        :param currentData: True if condition satisfies else False
        :return:
        """
        if 30 < currentData['RSI'] < 70:
            return True
        return False

    def __checkEmaSignalBuy(self, currentData):
        """
        Checks if the Close value is less than current data EMA
        :param currentData:
        :return: True if condition satisfies else False
        """
        if currentData['Close'] < currentData['EMA']:
            return True

        return False

    def __checkEmaSignalSell(self, currentData):
        """
        Checks if the Close value is greater than current data EMA
        :param currentData:
        :return: True if condition satisfies else False
        """
        if currentData['Close'] > currentData['EMA']:
            return True

        return False

    def getPreprocessedHistoricalData(self):
        """
        Gets the processed data and returns from 14th data point, as the RSI and EMA calculation is from 14th data point
        :return:
        """
        dataProcessor = HistoricalDataProcessor()
        dataProcessor.specifyHistoricalDataPath(path=self.__historicalDataPath)

        return dataProcessor.getProcessedData()[14:]

    def reset(self):
        print("\n<< Resetting >>")
        print(f"Sell reward: {self.sellReward} || Buy reward: {self.buyReward} || Hold reward: {self.buyReward}")

        self.current_step = 0
        self.__portfolio_value = self.__portfolioValue  # Reset the portfolio value
        self.__positions = 0
        self.__position_cost = 0
        self.__reward = 0
        self.__previousActions = [0]
        self.__done = False

        return self._get_observation()

    def __handleDeliveryTrades(self, current_data):

        # initializing a boolean variable which will indicate if the trade was executed or not
        executed = False
        currentMarketPrice = current_data['Close'] * self.__stockChunk

        # checking if it's a buy flag
        if self.__buyFlag:
            # doing common buy task
            self.__commonBuyTask(current_data=current_data)

            # setting the executed as true
            executed = True
        # if buy flag is not true but the positional cost is greater than current market price + half of current market
        # price and number of positions are less than 5
        elif not self.__buyFlag and self.__position_cost > (
                currentMarketPrice + currentMarketPrice * 0.5) and self.__positions < 5:  # adding a 0.5 buffer

            # checking if it has portfolio to buy
            if self.__portfolio_value >= current_data['Close']:
                self.__portfolio_value -= currentMarketPrice
                self.__position_cost += currentMarketPrice
                self.__positions += 1
                self.__reward = 30
                # setting the executed to True
                executed = True
            else:
                # if fund is insufficient giving it a negative reward
                self.__reward = -100
        else:
            # if it is not a buy flag giving a negative reward
            self.__reward = -100
            self.__done = True

        return executed

    def __handleDeliverySaleTask(self, sell_price):
        # if sell flag and there are existing positions
        if self.__sellFlag and self.__positions > 0:

            # checking if the sell price is greater than the hold position cost
            if sell_price > self.__position_cost:
                # calculating the profit
                self.__profit = sell_price - self.__position_cost

                # rewarding with the profit
                self.__reward = 100
                self.__portfolio_value += sell_price
            else:
                # calculating the loss
                self.__loss = self.__position_cost - sell_price
                self.__portfolio_value += sell_price

                # rewarding with negative loss x 2
                self.__reward = -100

            self.__positions = 0
            self.__position_cost = 0
            self.__sellFlag = False
            executed = True

        # checking if not sell flag and there are positions and the sell price is greater than position cost +
        # position cost * 0.5 (Buffer value)
        elif not self.__sellFlag and self.__positions > 0 and sell_price > self.__position_cost + (
                self.__position_cost * 0.5):  # setting a buffer of 0.5
            self.__profit = sell_price - self.__position_cost
            self.__reward = 70
            self.__portfolio_value += sell_price
            self.__positions = 0
            self.__position_cost = 0
            self.__sellFlag = False
            executed = True

        else:
            executed = False
            self.__reward = -70

        return executed

    def __handleBuyAction(self, current_data):

        executed = True
        self.__profit = 0

        """
            If trade type is delivery handling that and setting the buy and sell flag to False
            Otherwise executing common buy tasks
        """
        if self.__tradeTypeDelivery:
            executed = self.__handleDeliveryTrades(current_data=current_data)
            self.__buyFlag = False
            self.__sellFlag = False
        else:
            self.__commonBuyTask(current_data=current_data)

        return executed

    def __handleSellAction(self, current_data):
        self.__profit = 0
        self.__loss = 0
        executed = True
        sell_price = current_data['Close'] * (self.__positions * self.__stockChunk)

        if self.__positions == 0:
            self.__reward = -0.2
            return sell_price, executed

        # checking if trade type is delivery
        if self.__tradeTypeDelivery:
           executed = self.__handleDeliverySaleTask(sell_price=sell_price)

        else:
            # for Intraday cases common sell tasks
            if sell_price > self.__position_cost:
                self.__profit = sell_price - self.__position_cost

                self.__reward = 1

            else:
                self.__loss = self.__position_cost - sell_price

                self.__reward = -1

            self.__position_cost = 0
            self.__portfolio_value += sell_price

            self.__positions = 0

        return sell_price, executed

    def __commonBuyTask(self, current_data):
        # getting the current position value
        currentPositionValue = current_data['Close'] * self.__stockChunk

        previousPositionCost = self.__previousPositionCost

        self.__previousPositionCost = currentPositionValue
        self.__positions += 1

        if self.__positions >= 2:
            # checking if the position cost is greater than twice the current positions value
            if previousPositionCost >= currentPositionValue:
                # checking if there are more than 4 positions
                if self.__positions >= 5:
                    # negative rewards for buying again
                    self.__reward = 0.3
                # if there are less than 5 positions
                else:
                    # setting the rewards to positions x stock chunk
                    self.__reward = 0.6

            # checking if positions cost is less than current position value
            elif previousPositionCost <= currentPositionValue:

                # giving a negative reward of the value of the current price - the position cost we have
                self.__reward = -0.6
        else:
            self.__reward = -0.2

        # subtracting the current position value from the portfolio value
        self.__portfolio_value -= currentPositionValue

        # adding the current position value to the position cost
        self.__position_cost += currentPositionValue

    def __handleHoldAction(self, current_data):
        currentPositionValue = current_data['Close'] * (self.__positions * self.__stockChunk)

        # if there are no positions but the action is hold rewarding negative 1
        if self.__positions == 0:
            self.__reward = -1

        # checking if there was a possibility of profit
        elif currentPositionValue > self.__position_cost:
            self.__reward = - 0.8

        # checking if there was a possibility of loss
        elif currentPositionValue < self.__position_cost:
            self.__reward = 0.8

        # checking if the signals are saying to hold
        if self.__checkRsiSignalHold(currentData=current_data) and self.__reward > 0:
            self.__reward += self.__reward * 0.2
        else:
            # self.__reward -= self.__reward * 0.3
            pass

    def step(self, action):

        self.current_step += 1

        self.__profit = 0
        self.__loss = 0
        self.__reward = 0

        # if trade type delivery and buy and sell flags are False
        if self.__tradeTypeDelivery and not self.__buyFlag and not self.__sellFlag:
            self.__deliveryPeriodCount += 1

        # checking if delivery period count is same as delivery period then setting the buy and sell flag to True
        # and resetting the delivery period count
        if self.__tradeTypeDelivery and self.__deliveryPeriodCount == self.__deliveryPeriod:
            self.__buyFlag = True
            self.__sellFlag = True
            self.__deliveryPeriodCount = 0

        executed = True

        self.__action = action
        self.__previousActions.append(self.__action)

        # Get the current market data
        current_data = self.data.iloc[self.current_step]

        # if self.current_step + 1 == len(self.data) or self.__portfolio_value < current_data['Close']:
        if self.current_step + 1 == len(self.data) or self.__portfolio_value < current_data['Close'] * self.__stockChunk:
            self.__done = True

        if self.__done:
            self.__reward -= 10
        else:
            self.__reward += 5

        # TODO: remove
        sellValue = 0
        # Take action based on the agent's decision
        if action == 0:  # Sell action
            sellValue, executed = self.__handleSellAction(current_data=current_data)

            # if rsi signal for SELL satisfies rewarding 2x
            if self.__checkRsiSignalSell(currentData=current_data):
                if self.__reward > 0:
                    self.__reward += self.__reward * 0.2
            else:
                # else doing half of the reward
                # self.__reward -= self.__reward * 0.3
                pass

            # if EMA signal for SELL satisfies rewarding 2x
            if self.__checkEmaSignalSell(currentData=current_data):
                if self.__reward > 0:
                    self.__reward += self.__reward * 0.2
            else:
                # else doing 30 percent of the reward
                # self.__reward -= self.__reward * 0.3
                pass

            self.sellReward += self.__reward

        elif action == 1:  # Buy action
            executed = self.__handleBuyAction(current_data=current_data)

            # if rsi signal for BUY satisfies rewarding 2x
            if self.__checkRsiSignalBuy(currentData=current_data):
                self.__reward += self.__reward * 0.2
            else:
                # else doing half of the reward
                # self.__reward -= self.__reward * 0.3
                pass

            # if EMA signal for BUY satisfies rewarding 2x
            if self.__checkEmaSignalBuy(currentData=current_data):
                self.__reward += self.__reward * 0.2
            else:
                # else doing 30 percent of the reward
                # self.__reward -= self.__reward * 0.3
                pass

            self.buyReward += self.__reward

        elif action == 2:  # Hold
            self.__handleHoldAction(current_data=current_data)
            self.holdReward += self.__reward

        if self.__tradeTypeDelivery:
            info = {'portfolio_value': self.__portfolio_value, 'profit': self.__profit, 'position': self.__positions,
                    'bank': self.bank,
                    'position_cost': self.__position_cost, 'sellValue': sellValue, 'loss': self.__loss,
                    'deliveryPeriodCount': self.__deliveryPeriodCount, 'executed': executed, 'reward': self.__reward}
        else:
            info = {'portfolio_value': self.__portfolio_value, 'profit': self.__profit, 'position': self.__positions,
                    'bank': self.bank,
                    'position_cost': self.__position_cost, 'sellValue': sellValue, 'loss': self.__loss,
                    'reward': self.__reward}

        # if self.__action == 0:
        #     print(f"Action: Sell || Reward: {self.__reward} || Profit: {self.__profit} || Loss: {self.__loss} || Positions : {self.__positions} || Position cost: {self.__position_cost}\n")
        # elif self.__action == 1:
        #     print(f"Action: Buy || Reward: {self.__reward} || Profit: {self.__profit} || Loss: {self.__loss} || Positions : {self.__positions} || Position cost: {self.__position_cost}\n")
        # else:
        #     print(f"Action: Hold || Reward: {self.__reward} || Positions : {self.__positions} || Position cost: {self.__position_cost} || Current market price: { current_data['Close'] * (self.__positions * self.__stockChunk)}\n")

        return self._get_observation(), self.__reward, self.__done, info

    def _get_observation(self):

        # Retrieve the OHLCV, RSI, and EMA values for the current step
        closePrice = self.data.iloc[self.current_step]['Close']
        rsi = self.data.iloc[self.current_step]['RSI']
        ema = self.data.iloc[self.current_step]['EMA']

        observationItems = [closePrice, rsi, ema, self.__profit, self.__loss, self.__positions, self.__position_cost,
                            self.__previousActions[-1], self.__portfolioValue]

        scaledObservationItems = self.scaler.fit_transform([[item] for item in observationItems])
        scaledObservationItems = [item[0] for item in scaledObservationItems]

        # Combine the scaled values into an observation array
        observation = np.array(scaledObservationItems)

        observation = np.array(observation)
        return observation
