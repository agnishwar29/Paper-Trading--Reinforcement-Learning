import numpy as np
import ta
import pandas as pd


class HistoricalDataProcessor:

    def __init__(self):
        self.__historicalData = None

    def specifyHistoricalDataPath(self, path):

        self.__historicalData = pd.read_csv(path)

    def __changeDatetime(self):
        self.__historicalData['Date'] = pd.to_datetime(self.__historicalData['Date'], format="%Y-%m-%d")

    def __stripData(self):
        # self.__historicalData = self.__historicalData[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        self.__historicalData = self.__historicalData[['Open', 'High', 'Low', 'Close', 'Volume']]

    def __removeNanValues(self):

        self.__historicalData.dropna(inplace=True)

    def __calculateRSI(self, period):
        delta = self.__historicalData["Close"].diff()

        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        _gain = up.ewm(com=(period - 1), min_periods=period).mean()
        _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()

        RS = _gain / _loss

        self.__historicalData['RSI'] = pd.Series(100 - (100 / (1 + RS)), name="RSI")

    def getProcessedData(self):

        # stripping the historical data
        self.__stripData()

        # removing nan values from the dataset
        self.__removeNanValues()

        # calculating the RSI
        self.__calculateRSI(period=14)

        return self.__historicalData

