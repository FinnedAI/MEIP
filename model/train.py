from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import os
import pickle
from datetime import datetime

if not os.path.exists("saves/crude_embeddings.pkl"):
    raise ValueError("Please run --process first to generate embeddings")
else:
    with open("saves/crude_embeddings.pkl", "rb") as f:
        crude_embeddings = pickle.load(f)


def normalize_embeddings():
    middle_data = {}
    for ticker, date in crude_embeddings.keys():
        if ticker not in middle_data:
            middle_data[ticker] = {}
        middle_data[ticker][date] = crude_embeddings[(ticker, date)]

    for ticker in middle_data:
        # sort the dates in ascending order
        sorted_dates = sorted(
            middle_data[ticker], key=lambda x: datetime.strptime(x, "%Y-%m-%d")
        )

        # create a new dictionary with sorted dates
        sorted_data = {date: middle_data[ticker][date] for date in sorted_dates}

        # replace the original dictionary with the sorted one
        middle_data[ticker] = sorted_data

    normalized_embeddings = []
    for ticker in middle_data.keys():
        for date in list(middle_data[ticker].keys())[:-30]:
            normalized_embeddings.append(middle_data[ticker][date])

    normalized_embeddings = normalize(normalized_embeddings)

    embeddings_keys = {}
    i = 0
    for ticker in tqdm(middle_data.keys(), desc="Creating embeddings keys"):
        for date in list(middle_data[ticker].keys())[:-30]:
            embeddings_keys[tuple(normalized_embeddings[i])] = (ticker, date)
            i += 1

    return normalized_embeddings, embeddings_keys


def knn(normalized_embeddings, embeddings_keys):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
    save = [nbrs, normalized_embeddings, embeddings_keys]
    # Save the model to disk
    with open("saves/knn.pkl", "wb") as f:
        pickle.dump(save, f)
