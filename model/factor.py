import datetime

import numpy as np
import pandas as pd

import yfinance as yf

from sklearn.linear_model import LinearRegression

class Factor:
    def __init__(self, data, ticker):
        self.data = data
        self.index = data.index
        self.ticker = ticker
        self.daily_ret = data['Close'].pct_change() # from ytd to tdy

    def get_risk_free_rate(self):
        """
        Request the risk-free rate from yfinance by 3-month T-bill.

        Returns
        -------
        float
            The risk-free rate.
        """
        # search for the IRX.csv first in the data folder
        irx_path = f"data/yfinance/IRX.csv"
        if os.path.exists(irx_path):
            irx_data = pd.read_csv(irx_path, index_col=0, parse_dates=True)
            return irx_data
        else:
            start = self.index[0]
            end = pd.Timestamp(self.index[-1]) + pd.Timedelta(days=1)
            annualized_risk_free_rate = yf.download("^IRX", auto_adjust=True, start=start, end=end)["Close"]
            # Turn annualized risk-free rate into daily
            daily_risk_free_rate = annualized_risk_free_rate.apply(lambda x: (1 + x / 100) ** (1 / 365) - 1)
            daily_risk_free_rate.to_csv(irx_path)
            return daily_risk_free_rate

    def add_momentum(self, window=10, ex_curr=0, vol_adj=True, excess=False, risk_free_rate=None):
        """
        Adds momentum factor to the data.

        Parameters
        ----------
        window : int
            The window size for calculating momentum.
        ex_curr : int
            The number of periods to shift for the current price.
        vol_adj : bool
            Whether to adjust for volatility.
        excess : bool
            Whether to calculate excess returns.
        risk_free_rate : float, array-like or None
            The risk-free rate to use for excess return calculation.
            If None, the risk-free rate will be requested from yfinance.

        Returns
        -------
        None
        """
        # Calculate the return from years ago, excluding [ex_curr] days return
        ret = (self.data['Close'].shift(ex_curr) - self.data['Close'].shift(window)) / self.data['Close'].shift(window)
        
        if excess:
            if risk_free_rate is None:
                risk_free_rate = self.get_risk_free_rate()
            # Turn return to excess return
            ret = ret - risk_free_rate["^IRX"]
        
        if vol_adj:
            # Adjust for volatility
            sigma = self.daily_ret.shift(ex_curr).rolling(window - ex_curr).std()
            self.data['MOM_vol_adj'] = ret / sigma
        else:
            self.data['MOM'] = ret

    def add_skewness(self, window=63, method='base'):
        """
        Adds skewness factor to the data.

        Parameters
        ----------
        window : int
            The window size for calculating skewness.
        method : str = ['base', 'opt_implied', 'opt_proxy']
            The method to use for calculating skewness.
            Note that 'opt_implied' and 'opt_proxy' haven't been implemented yet.

        Returns
        -------
        None
        """
        if method == 'base':
            self.data['Skewness'] = self.daily_ret.rolling(window).apply(lambda x: pd.Series(x).skew(), raw=False)
        elif method == 'opt_implied':
            print("Option implied skewness calculation not implemented yet.")
        elif method == 'opt_proxy':
            print("Option proxy skewness calculation not implemented yet as data is not available.")
            # ticker = yf.Ticker(self.ticker)
            # expires = ticker.options
            # expires = [datetime.datetime.strptime(x, '%Y-%m-%d').date() for x in expires]
            # target_date = datetime.date.today() + datetime.timedelta(days=30)
            
            # K = self.data['Close'].iloc[-1]
            # cols = ["strike", "impliedVolatility", "inTheMoney"]
            # if target_date in expires:
            #     opt = ticker.option_chain(str(target_date))
            #     opt_calls = opt.calls[opt.calls["inTheMoney"] == False][cols]
            #     opt_puts = opt.puts[opt.puts["inTheMoney"] == False][cols]
            #     data = pd.concat([opt_calls, opt_puts]).drop(columns=["inTheMoney"])
            #     data["log_moneyness"] = np.log(data["strike"] / K)
            #     X, y = data["log_moneyness"], data["impliedVolatility"]
            #     model = LinearRegression()
            #     model.fit(X.values.reshape(-1, 1), y)
            #     self.data['SKW_opt_proxy'] = model.coef_[0]
            # else:
            #     closest_exp = str(min(expires, key=lambda x: abs(x - target_date)))
            #     second_closest_exp = str(min(
            #         (x for x in expires if x != closest_exp),
            #         key=lambda x: abs(x - target_date),
            #         default=None
            #     ))

    def add_volatility(self, window=63, method='base'):
        """
        Adds volatility factor to the data.

        Parameters
        ----------
        window : int
            The window size for calculating volatility.
        method : str = ['base', 'normalized']
            The method to use for calculating volatility.

        Returns
        -------
        None
        """
        if method == 'base':
            self.data['Volatility'] = self.daily_ret.rolling(window=window).std()
        elif method == 'normalized':
            self.data['Volatility'] = self.daily_ret.rolling(window=window).std() / self.daily_ret.rolling(window=window).mean()

    def add_liquidity(self, method='base'):
        """
        Adds liquidity factor to the data.

        Parameters
        ----------
        method : str = ['base']
            The method to use for calculating liquidity.

        Returns
        -------
        None
        """
        start = self.index[0]
        end = pd.Timestamp(self.index[-1]) + pd.Timedelta(days=1)
        # Download outstanding shares
        os_shares = yf.Ticker(self.ticker).get_shares_full(start=start, end=end)
        os_shares = os_shares.astype(float)
        os_shares.index = os_shares.index.date
        # Take the dates that with non-zero value at column "Stock Splits"
        splits_dates = self.data[self.data['Stock Splits'] != 0].index
        for date in splits_dates:
            # Convert date to datetime.date if needed
            if isinstance(date, pd.Timestamp):
                date_obj = date.date()
            else:
                date_obj = date
            # Adjust outstanding shares for all dates before and including the split date
            os_shares.loc[os_shares.index <= date_obj] *= self.data.loc[date]['Stock Splits']
        # Remove duplicates (keep the last one)
        os_shares = os_shares[~os_shares.index.duplicated(keep='last')]
        # Forward fill missing values to align the index
        os_shares = os_shares.reindex(self.data.index, method='ffill')

        self.data['Liquidity'] = self.data["Volume"] / os_shares

    def add_value(self, method='B/M'):
        """
        Adds value factor to the data.

        Parameters
        ----------
        method : str = ['B/M', 'E/P']
            The method to use for calculating value.
            Note that yfinance don't provide historical balance sheets or outstanding shares.
            I instead download them from alphavantage API.
            Please see https://www.alphavantage.co/documentation/ for more information.
            The E/P ratio hasn't been implemented yet.

        Returns
        -------
        None
        """
        if method == 'B/M':
            # Read the balance sheet data
            BS = pd.read_csv(f'../../data/alphavantage/{self.ticker}_balance_sheet.csv', index_col='fiscalDateEnding', parse_dates=True)
            # Reverse and reindex to align to the data
            Equity = BS['totalShareholderEquity'][::-1]
            Equity = Equity.reindex(self.data.index, method='ffill')

            start = self.index[0]
            end = pd.Timestamp(self.index[-1]) + pd.Timedelta(days=1)
            # Download outstanding shares
            os_shares = yf.Ticker(self.ticker).get_shares_full(start=start, end=end)
            os_shares = os_shares.astype(float)
            os_shares.index = os_shares.index.date
            # Take the dates that with non-zero value at column "Stock Splits"
            splits_dates = self.data[self.data['Stock Splits'] != 0].index
            for date in splits_dates:
                # Convert date to datetime.date if needed
                if isinstance(date, pd.Timestamp):
                    date_obj = date.date()
                else:
                    date_obj = date
                # Adjust outstanding shares for all dates before and including the split date
                os_shares.loc[os_shares.index <= date_obj] *= self.data.loc[date]['Stock Splits']
            # Remove duplicates (keep the last one)
            os_shares = os_shares[~os_shares.index.duplicated(keep='last')]
            # Forward fill missing values to align the index
            os_shares = os_shares.reindex(self.data.index, method='ffill')

            self.data['Value_BM'] = (Equity / os_shares) / self.data['Close']
        # elif method == 'E/P':
        #     self.data['Value_EP'] = self.data['Earnings'] / self.data['Price']

    def add_market_cap(self, method='base'):
        """
        Adds market capitalization factor to the data.

        Parameters
        ----------
        method : str = ['base', 'log']
            The method to use for calculating market capitalization.

        Returns
        -------
        None
        """
        start = self.index[0]
        end = pd.Timestamp(self.index[-1]) + pd.Timedelta(days=1)
        # Download outstanding shares
        os_shares = yf.Ticker(self.ticker).get_shares_full(start=start, end=end)
        os_shares = os_shares.astype(float)
        os_shares.index = os_shares.index.date
        # Take the dates that with non-zero value at column "Stock Splits"
        splits_dates = self.data[self.data['Stock Splits'] != 0].index
        for date in splits_dates:
            # Convert date to datetime.date if needed
            if isinstance(date, pd.Timestamp):
                date_obj = date.date()
            else:
                date_obj = date
            # Adjust outstanding shares for all dates before and including the split date
            os_shares.loc[os_shares.index <= date_obj] *= self.data.loc[date]['Stock Splits']
        # Remove duplicates (keep the last one)
        os_shares = os_shares[~os_shares.index.duplicated(keep='last')]
        # Forward fill missing values to align the index
        os_shares = os_shares.reindex(self.data.index, method='ffill')
        if method == 'base':
            self.data['Market_Cap'] = self.data['Close'] * os_shares
        elif method == 'log':
            self.data['Market_Cap'] = np.log(self.data['Close'] * os_shares)