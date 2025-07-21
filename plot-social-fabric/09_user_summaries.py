# %%
import pandas as pd
from pathlib import Path
import networkx as nx
from sklearn.decomposition import PCA
import numpy as np
from bokeh.plotting import figure, from_networkx, show
from bokeh.models import (
    HoverTool,
    ColumnDataSource,
    Circle,
    MultiLine,
    LabelSet,
    GraphRenderer,
    StaticLayoutProvider,
)

# from bokeh.palettes import Spectral8
import graspologic
import holoviews as hv
from holoviews import opts
import holoviews.operation.datashader as hd
import panel as pn
from node2vec import Node2Vec  # new import for node embeddings

import re
from typing import Dict, List, Optional, TypedDict
from toolz import curry, compose, partition_all
from collections import defaultdict, deque
from datetime import datetime

import modal
import json
from tqdm import tqdm

# Add UMAP and HDBSCAN imports
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer


def load_embeddings(filepath: str) -> dict:
    """Load embeddings from .npy file and metadata from .json file"""
    filepath = Path(filepath)

    # Load embeddings
    embeddings = np.load(filepath.with_suffix(".npy"))

    # Load metadata
    metadata_path = filepath.with_suffix("_metadata.json")
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        metadata["embeddings"] = embeddings
        return metadata
    else:
        return {"embeddings": embeddings}


def load_results(filepath: str) -> dict:
    """Load results from file - auto-detects format"""
    filepath = Path(filepath)

    # Try .npy format first (embeddings + metadata)
    if filepath.with_suffix(".npy").exists():
        return load_embeddings(filepath)

    # Try .pkl format
    elif filepath.with_suffix(".pkl").exists():
        import pickle

        with open(filepath.with_suffix(".pkl"), "rb") as f:
            return pickle.load(f)

    # Try .json format
    elif filepath.with_suffix(".json").exists():
        with open(filepath.with_suffix(".json"), "r") as f:
            return json.load(f)

    else:
        raise FileNotFoundError(
            f"No results file found for {filepath} (tried .npy, .pkl, .json)"
        )


def clean_tweet_text(text: str, max_length: int = 32000) -> str:
    text = re.sub(r"This Post is from a suspended account.*", "", text)
    text = re.sub(r"https://t\.co/\w+", "", text)
    text = re.sub(r"^RT @[A-Za-z0-9_]+: ", "", text)
    text = re.sub(r"^(\s*@\w+\s*)+", "", text)
    text = text.strip()
    return text[:max_length] + "..." if len(text) > max_length else text


# %%
# Load original dataframe for visualization
DATA_PATH = Path("../data/tweets_2023_11_week4.parquet")
df = pd.read_parquet(DATA_PATH)

"""
tweet_id	account_id	created_at	full_text	retweet_count	favorite_count	reply_to_tweet_id	reply_to_user_id	reply_to_username	archive_upload_id	updated_at	username	temporal_subset	topic
0	1686762764532920321	1.0021918513050583e+18	2023-08-02 15:35:56+00:00	@8chabard it does	0.0	2.0	1.6867615663378924e+18	52333946.0	8chabard	270.0	2025-01-25 18:49:16.732494+00:00	imhinesmi	True	Pharmaceutical Drugs
1	1346709099166121985	1.0021918513050583e+18	2021-01-06 06:44:04+00:00	@moonknower why not	0.0	1.0	1.346600877478568e+18	1.2522775117461832e+18	parafactual	270.0	2025-01-25 18:49:16.732494+00:00	imhinesmi	True	Space exploration
2	1327053481782349827	1.0021918513050583e+18	2020-11-13 00:59:40+00:00	@disconcerta whats the story?	0.0	2.0	1.327052842302124e+18	1.1143560472950784e+18	None	270.0	2025-01-25 18:49:16.732494+00:00	imhinesmi	True	Literature
3	1322302328158957568	1.0021918513050583e+18	2020-10-30 22:20:16+00:00	@grantadever its real fun	0.0	1.0	1.3223013339412234e+18	22525974.0	grantadever	270.0	2025-01-25 18:49:16.732494+00:00	imhinesmi	True	Environmental projects
4
"""

# %%
# group by

import json
from toolz import compose_left

# Define type for clarity
# UserRepr = dict[str, str]  # username -> representative string


#  group -> list
def get_top_tweets(group):
    return [
        f'{x["favorite_count"]} likes: {x["full_text"]}' for i, x in group.iterrows()
    ]


def get_top_tweets_str(group):
    return "\n\n".join(
        f"Tweet {i+1} (likes: {row['favorite_count']}): {row['full_text']}"
        for i, (_, row) in enumerate(group.iterrows())
    )


def filter_top_level_tweets(df):
    """Filter to only top-level tweets (no replies, retweets, or tweets with images)"""
    return df[
        df["reply_to_tweet_id"].isna()
        & ~df["full_text"].str.startswith("RT @", na=False)
        & ~df["full_text"].str.contains("https://t.co/", na=False)
    ]


process = compose_left(
    filter_top_level_tweets,
    lambda d: d.sort_values("favorite_count", ascending=False)
    .groupby("username")
    .head(5),
    lambda d: d.groupby("username")
    .apply(get_top_tweets, include_groups=False)
    .to_dict(),
    lambda d: {k: v for k, v in d.items() if len(v) >= 5},
    lambda d: {k: [clean_tweet_text(tweet, 500) for tweet in v] for k, v in d.items()},
    lambda d: {k: "\n\n".join(v) for k, v in d.items()},
)

user_top_tweets = process(df)

# with open("user_representative_strings.json", "w") as f:
#     json.dump(user_top_tweets, f)

# print(f"Generated representative strings for {len(user_top_tweets)} users")
# print("Saved to user_representative_strings.json")

# %%
from pprint import pprint

# sample five and print their tweets
pprint(user_top_tweets)
print(f"got {len(user_top_tweets)} users")


# %%
# Create TF-IDF vectors for all users' tweets
def create_user_tfidf_vectors(df, min_tweets=10):
    """Create TF-IDF vectors for each user based on all their tweets"""
    # Filter to top-level tweets
    filtered_df = filter_top_level_tweets(df)

    # Group by username and concatenate all tweets
    user_documents = (
        filtered_df.groupby("username")["full_text"]
        .apply(
            lambda tweets: " ".join(clean_tweet_text(tweet, 32000) for tweet in tweets)
        )
        .to_dict()
    )

    # Filter users with minimum tweet count
    tweet_counts = filtered_df.groupby("username").size()
    user_documents = {
        user: doc
        for user, doc in user_documents.items()
        if tweet_counts[user] >= min_tweets
    }

    print(f"Creating TF-IDF vectors for {len(user_documents)} users")

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
    )

    usernames = list(user_documents.keys())
    documents = list(user_documents.values())

    tfidf_matrix = vectorizer.fit_transform(documents)

    return {
        "usernames": usernames,
        "tfidf_matrix": tfidf_matrix,
        "vectorizer": vectorizer,
        "user_documents": user_documents,
    }


# Create TF-IDF vectors
tfidf_data = create_user_tfidf_vectors(df, min_tweets=10)
print(f"TF-IDF matrix shape: {tfidf_data['tfidf_matrix'].shape}")


# %%
