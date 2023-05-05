import nltk
import os
import csv
import yfinance as yf
import numpy as np
from datetime import datetime
import pickle
import pandas as pd
from tqdm import tqdm
import requests

# Sentiment analysis deps
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")  # Install only at first run
vader = SentimentIntensityAnalyzer()


if not os.path.exists("saves"):
    os.makedirs("saves")

if not os.path.exists("../raw_partner_headlines.csv"):
    # download from https://finned.tech/assets/data/raw_partner_headlines.csv
    print("Downloading raw data...")
    url = "https://finned.tech/assets/data/raw_partner_headlines.csv"
    r = requests.get(url, allow_redirects=True)
    open("../raw_partner_headlines.csv", "wb").write(r.content)


# Read the raw csv data, and return as dict with just the attrs we need
def make_data():
    data = {}
    with open("../raw_partner_headlines.csv", encoding="utf-8", newline="") as f:
        raw_csv = csv.DictReader(f)
        # apply preprocessing to normalize data format and names
        for row in tqdm(raw_csv, desc="Processing news data"):
            if row["stock"] not in data.keys():
                data[row["stock"]] = []

            data[row["stock"]].append(
                {
                    "headline": row["headline"],
                    "date": datetime.strptime(
                        row["date"].split(" ")[0], "%Y-%m-%d"
                    ).date(),
                    "sentiment": vader.polarity_scores(row["headline"])["compound"],
                }
            )

    return data


def get_stock_history(data):
    stocks = " ".join(list(data.keys()))
    ticker_data = yf.download(
        stocks, start="2010-02-03", end="2020-06-04"
    )  # data range is from 02-2010 to 06-2020
    return ticker_data


def make_embeddings_data(data, ticker_data):
    embeddings = {}

    for heading, ticker in tqdm(
        ticker_data.keys(), desc="Creating embeddings for tickers"
    ):
        if ticker not in embeddings:
            embeddings[ticker] = {}
        for date in ticker_data[(heading, ticker)].index:
            if heading not in ["Open", "Adj Close", "Volume"]:
                continue
            value = ticker_data[(heading, ticker)][date]
            if np.isnan(value):
                continue
            comparison_date = date.to_pydatetime().date()
            for row in data[ticker]:
                if row["date"] == comparison_date:
                    if str(comparison_date) not in embeddings[ticker]:
                        embeddings[ticker][str(comparison_date)] = {
                            "sentiment": row["sentiment"]
                        }
                    embeddings[ticker][str(comparison_date)][heading] = value

    return embeddings


def make_crude_embeddings(embeddings):
    crude_embeddings = {}

    for ticker in tqdm(embeddings.keys(), desc="Filling in missing data"):
        # 252 trading days in most years
        if len(embeddings[ticker].keys()) < 282:  # 252 trading days plus 30 days for predictions
            continue
        for date in list(embeddings[ticker].keys())[252:]:
            open_prc = embeddings[ticker][date]["Open"]
            close_prc = embeddings[ticker][date]["Adj Close"]
            sentiment = embeddings[ticker][date]["sentiment"]
            volume = embeddings[ticker][date]["Volume"]
            intra = (close_prc - open_prc) / open_prc

            # calculate volatility YTD
            closes = [
                embeddings[ticker][row]["Adj Close"]
                for row in list(embeddings[ticker].keys())[:252]
            ]
            returns = pd.Series(closes).pct_change()
            volatility = returns.std() * (252**0.5)
            crude_embeddings[(ticker, date)] = [
                open_prc,
                close_prc,
                volume,
                intra,
                volatility,
                sentiment,
            ]

    # Save crude embeddings to file
    with open("saves/crude_embeddings.pkl", "wb") as f:
        pickle.dump(crude_embeddings, f)

    return crude_embeddings
