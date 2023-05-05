import argparse
import process
import train
import infer
import utils
from matplotlib import pyplot as plt
import re

args = argparse.ArgumentParser()
args.add_argument("--process", action="store_true", help="Process data")
args.add_argument("--make-embedding", type=str, help="Make embedding for ticker")
args.add_argument("--train", action="store_true", help="Train model on embeddings")
# infer, taking as input a list like "1 2 3 4 5"
args.add_argument("--infer", type=str, help="Infer model")
opts = args.parse_args()

if __name__ == "__main__":
    if opts.process:
        print("Processing data...")
        data = process.make_data()
        ticker_data = process.get_stock_history(data)
        embeddings = process.make_embeddings_data(data, ticker_data)
        crude_embeddings = process.make_crude_embeddings(embeddings)

    elif opts.make_embedding:
        print(f"Making embedding for {opts.make_embedding}...")

        emb_args = {}
        headline = None

        pattern = r"headline='(?P<headline>[^']*)'.*date=(?P<date>\d{4}-\d{2}-\d{2}).*ticker=(?P<ticker>[A-Za-z\-]+)"

        match = re.search(pattern, opts.make_embedding)
        if not match.group("ticker"):
            raise ValueError("Please specify the ticker")

        headline = match.group("headline")
        date = match.group("date")
        ticker = match.group("ticker")

        out_embedding = utils.make_embedding(ticker, headline, date)
        print(f"Embedding: {out_embedding}")

    elif opts.train:
        print("Training KNN model...")

        normalized, keys = train.normalize_embeddings()
        train.knn(normalized, keys)
        print("Model saved to saves/knn.pkl")

    elif opts.infer:
        print("Inferring model...")
        infer_args = {}
        for arg in opts.infer.split():
            key, value = arg.split("=")
            infer_args[key] = value

        embedding = infer_args.get("embedding")
        days = infer_args.get("days")

        if not embedding or not days:
            raise ValueError("Please specify model, embedding, and days")

        print("Predicting KNN model...")

        series = infer.knn(embedding.split(","), int(days))
        plt.plot(series)
        # x is the number of days
        plt.xlabel("Days")
        # y is the Adj Close price
        plt.ylabel("Adj Close")
        plt.show()

    else:
        raise ValueError("Please specify an action")
