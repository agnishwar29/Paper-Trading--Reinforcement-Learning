from sklearn.preprocessing import MinMaxScaler
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

    def __scaleValues(self):
        scaler = MinMaxScaler()
        columns = self.__historicalData.columns
        self.__historicalData = pd.DataFrame(scaler.fit_transform(self.__historicalData), columns=columns)

    def __calculateRSI(self, period):
        delta = self.__historicalData["Close"].diff()

        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        _gain = up.ewm(com=(period - 1), min_periods=period).mean()
        _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()

        RS = _gain / _loss

        self.__historicalData['RSI'] = pd.Series(100 - (100 / (1 + RS)), name="RSI")

    def __calculateEma(self, period):
        self.__historicalData['EMA'] = self.__historicalData['Close'].ewm(span=period, adjust=False).mean()

    def getProcessedData(self):

        # stripping the historical data
        self.__stripData()

        # removing nan values from the dataset
        self.__removeNanValues()

        # change date format
        # self.__changeDatetime()

        # calculating the RSI
        self.__calculateRSI(period=14)

        self.__calculateEma(period=14)

        # scale data
        # self.__scaleValues()

        return self.__historicalData

