import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime as dt
import pandas as pd

vader = SentimentIntensityAnalyzer()


def make_embedding(ticker, headline=None, date=None):
    if headline:
        sentiment = vader.polarity_scores(headline)["compound"]
    else:
        sentiment = 0

    end = dt.datetime.strptime(date, "%Y-%m-%d").date() if date else dt.datetime.now()
    start = end - dt.timedelta(days=3650)

    ticker_data = yf.download(ticker, start=start, end=end)
    # get the last row of the dataframe's Adj Close, Open, and Volume columns
    # and assign them to variables
    open_prc = ticker_data["Open"][-1]
    close_prc = ticker_data["Adj Close"][-1]
    volume = ticker_data["Volume"][-1]
    intra = (close_prc - open_prc) / open_prc

    closes = [
        ticker_data["Adj Close"][-i]
        for i in range(1, len(ticker_data["Adj Close"]) + 1)
    ]
    returns = pd.Series(closes).pct_change()
    volatility = returns.std() * (252**0.5)  # annualized volatility

    return ",".join(
        [
            str(open_prc),
            str(close_prc),
            str(volume),
            str(intra),
            str(volatility),
            str(sentiment),
        ]
    )
