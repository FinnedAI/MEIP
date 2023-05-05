import os
import pickle
from sklearn.preprocessing import normalize
from random import choices
from tqdm import trange
import numpy as np
from datetime import datetime
import hyperparams as hp

if not os.path.exists("saves/crude_embeddings.pkl"):
    raise ValueError("Please run --process first to generate embeddings")
else:
    with open("saves/crude_embeddings.pkl", "rb") as f:
        crude_embeddings = pickle.load(f)
        middle_data = {}
        for ticker, date in crude_embeddings.keys():
            if ticker not in middle_data:
                middle_data[ticker] = {}
            middle_data[ticker][date] = crude_embeddings[(ticker, date)]


def knn(embedding, days):
    if not os.path.exists("saves/knn.pkl"):
        raise ValueError("Please run --train knn first to generate model")

    with open("saves/knn.pkl", "rb") as f:
        save = pickle.load(f)
        nbrs = save[0]
        normalized_embeddings = save[1]
        embeddings_keys = save[2]

    closes = [float(embedding[1])]

    def inner(embedding):
        # find neighbor for embedding
        embedding = normalize([embedding])

        # we have to refit the model every time because we want to penalize
        # the model for predicting the same ticker twice in a row
        nbrs.fit(normalized_embeddings)

        distances, indices = nbrs.kneighbors(embedding)
        # get the ticker and date of the indices
        value = normalized_embeddings[indices[0][0]]
        ticker, date = embeddings_keys[tuple(value)]
        # get the next day in the dataset for the ticker from middle_data
        tomorrow = list(middle_data[ticker].keys()).index(date)
        tomorrow = list(middle_data[ticker].keys())[tomorrow + 1]

        # get the Adj Close price of the next day
        try:
            tdy_close = crude_embeddings[(ticker, date)][1]
            tmr_close = crude_embeddings[(ticker, tomorrow)][1]
            pct_change = (tmr_close - tdy_close) / (
                tdy_close * hp.PCT_DULL
            )
            diff = (
                datetime.strptime(tomorrow, "%Y-%m-%d")
                - datetime.strptime(date, "%Y-%m-%d")
            )
            if diff.days > hp.MAX_DAYS_DIFF:
                pct_change = 0

            closes.append(closes[-1] * (1 + pct_change))
        except KeyError:
            pass

        return ticker, tomorrow

    # set fade rate to avoid predicting the same ticker over and over
    fades = {}
    sent_fade = 0  # We don't want a new sentiment for every sample, so we slowly decrease the effects of the existing one
    for i in trange(days, desc="Predicting Adj Close prices"):
        ticker, tomorrow = inner(embedding)
        embedding = crude_embeddings[(ticker, tomorrow)]
        # Set to random number (generally positive) between -1 and 1, where
        # we expect clustering around 0.
        if abs(sent_fade) < hp.SENT_ABS_MIN:
            total_prob = sum(hp.SENT_PROBS)
            norm_probs = [p / total_prob for p in hp.SENT_PROBS]
            sent_fade = choices(hp.SENT_PROB_DIST, norm_probs)[0] / 10
        else:
            sent_fade *= hp.SENT_FADE
        embedding[5] = sent_fade

        # penality for predicting the same ticker twice in a row
        normalized_embeddings = []
        for emb in embeddings_keys.keys():
            # This a new ticker, so we don't have a fade rate for it yet
            if ticker not in fades:
                fades[ticker] = hp.NEW_TICKER_FADE
            # This is a ticker from the previous prediction,
            # and we want to penalize it for being predicted twice in a row
            elif fades[ticker] <= hp.NEW_TICKER_FADE:
                fades[ticker] *= hp.TICKER_REPEATED_PENALTY

            # Reduce the fade rate for all tickers except the one we just predicted
            for t in fades.keys():
                if t != ticker:
                    fades[t] = min(fades[t] * hp.TICKER_RESTORE, 1)

            if embeddings_keys[emb][0] == ticker:
                normalized_embeddings.append(list(emb * np.array(fades[ticker])))
            else:
                normalized_embeddings.append(list(emb))

    return closes
