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

    def getProcessedData(self):

        # stripping the historical data
        self.__stripData()

        # removing nan values from the dataset
        self.__removeNanValues()

        # changing the string date to datetime format
        # self.__changeDatetime()

        return self.__historicalData

