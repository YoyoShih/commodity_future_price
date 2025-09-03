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
        self.daily_ret = data['Close'].pct_change()

    def get_risk_free_rate(self):
        """
        Request the risk-free rate from yfinance by 3-month T-bill.

        Returns
        -------
        float
            The risk-free rate.
        """
        start = self.index[0]
        end = pd.Timestamp(self.index[-1]) + pd.Timedelta(days=1)
        annualized_risk_free_rate = yf.download("^IRX", auto_adjust=True, start=start, end=end)["Close"]
        # Turn annualized risk-free rate into daily
        daily_risk_free_rate = annualized_risk_free_rate.apply(lambda x: (1 + x / 100) ** (1 / 365) - 1)
        return daily_risk_free_rate

    def add_momentum(self, window=252, ex_curr=0, vol_adj=True, excess=False, risk_free_rate=None):
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
        # Calculate the return
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

    def add_skewness(self, window=252, method='base'):
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

    def add_Volatility(self, window=252, method='base'):
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

    def add_Open_Interest(self, method='base'):
        """
        Adds open interest factor to the data.

        Parameters
        ----------
        method : str = ['base']
            The method to use for calculating open interest.

        Returns
        -------
        None
        """
        start = self.index[0]
        end = pd.Timestamp(self.index[-1]) + pd.Timedelta(days=1)
        # Download outstanding shares
        os_shares = yf.Ticker(self.ticker).get_shares_full(start=start, end=end)
        os_shares.index = os_shares.index.date
        # Remove duplicates (keep the last one)
        os_shares = os_shares[~os_shares.index.duplicated(keep='last')]
        # Forward fill missing values to align the index
        os_shares = os_shares.reindex(self.data.index, method='ffill')
        self.data['Op_Int'] = self.data["Volume"] / os_shares

    def add_Value(self, method='B/M'):
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
            os_shares.index = os_shares.index.date
            # Remove duplicates (keep the last one)
            os_shares = os_shares[~os_shares.index.duplicated(keep='last')]
            # Forward fill missing values to align the index
            os_shares = os_shares.reindex(self.data.index, method='ffill')

            self.data['Value_BM'] = (Equity / os_shares) / self.data['Close']
        # elif method == 'E/P':
        #     self.data['Value_EP'] = self.data['Earnings'] / self.data['Price']

    def add_inflation_beta(self, window=60, method='base'):
        """
        Adds inflation beta factor to the data.

        Parameters
        ----------
        window : int
            The window size for calculating inflation beta.
            Default to be 3 years.
        method : str = ['base']
            The method to use for calculating inflation beta.

        Returns
        -------
        None
        """
        if method == 'base':
            # Read the CPI data
            CPI = pd.read_csv('../../data/alphavantage/CPI.csv', index_col='date', parse_dates=True)[::-1]
            # Truncate to the relevant date range
            CPI = CPI.loc[self.index[0]:self.index[-1]]
            # Align the data index to CPI by nearest date
            month_data = self.data['Close'].reindex(CPI.index, method='nearest')
            # Calculate inflation beta
            inflation_beta = self.rolling_regression(month_data.pct_change(fill_method=None), CPI.pct_change(), window)
            # Re-align the index back
            self.data['Inflation_Beta'] = inflation_beta.reindex(self.data.index, method='ffill')

    def rolling_regression(self, X, y, window):
        """
        Performs rolling regression on the given data.

        Parameters
        ----------
        X : pd.Series
            The independent variable.
        y : pd.Series
            The dependent variable.
        window : int
            The window size for the rolling regression.

        Returns
        -------
        pd.Series
            The rolling regression coefficients.
        """
        # Create a DataFrame to hold the results
        results = pd.Series(index=X.index)
        X = X.dropna()
        y = y.dropna()

        # Perform rolling regression
        for i in range(window, len(X)):
            X_window = X[i-window:i]
            y_window = y[i-window:i]
            model = LinearRegression().fit(X_window.values.reshape(-1, 1), y_window)
            results.iloc[i] = model.coef_[0]

        return results