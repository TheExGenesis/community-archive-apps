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
from toolz import compose_left


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


# load embeddings from embedded_clustered_tweets

results = load_results("embedded_clustered_tweets")
print(f"Loaded {len(results['texts'])} texts")
print(f"Embeddings shape: {np.array(results['embeddings']).shape}")
print(f"Found {results['clustering']['n_clusters']} clusters")
# %%
# Create bokeh plot data
embeddings = np.array(results["embeddings"])

# %%
# Reduce dimensions to 5D using UMAP
print(f"Original embeddings shape: {embeddings.shape}")
print("Reducing dimensions to 5D using UMAP...")

umap_10d = umap.UMAP(
    n_components=10, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
)

embeddings_10d = umap_10d.fit_transform(embeddings)
print(f"5D embeddings shape: {embeddings_10d.shape}")


# %%
# Create 2D projection for visualization
print("Creating 2D projection for visualization...")

umap_2d = umap.UMAP(
    n_components=2, n_neighbors=30, min_dist=0.0, metric="cosine", random_state=42
)

embeddings_2d = umap_2d.fit_transform(embeddings)
print(f"2D embeddings shape: {embeddings_2d.shape}")
# %%
# Cluster using HDBSCAN
print("Clustering with HDBSCAN...")

# Auto-estimate clustering parameters based on dataset size
n_texts = len(embeddings_10d)
min_cluster_size = max(5, int(n_texts * 0.001))
min_samples = max(2, int(n_texts * 0.0005))
print(f"Min cluster size: {min_cluster_size}")
print(f"Min samples: {min_samples}")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    metric="euclidean",
    cluster_selection_method="eom",
)

cluster_labels = clusterer.fit_predict(embeddings_10d)
cluster_probabilities = clusterer.probabilities_

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"Found {n_clusters} clusters")
print(f"Noise points: {n_noise}")
print(
    f"Parameters used: min_cluster_size={min_cluster_size}, min_samples={min_samples}"
)

# %%
# Load original dataframe for visualization
DATA_PATH = Path("../data/tweets_2023_11_week4.parquet")
df = pd.read_parquet(DATA_PATH)

# Create color mapping for clusters with size based on log likes
unique_labels = list(set(cluster_labels))
base_colors = [f"#{i:06x}" for i in range(0, 0xFFFFFF, 0xFFFFFF // len(unique_labels))]
color_map = dict(zip(unique_labels, base_colors))

# Calculate sizes based on log of like count
favorite_counts = df["favorite_count"].tolist()

# Create size values based on log scale
sizes = []
for likes in favorite_counts:
    # Add 1 to avoid log(0), then scale to reasonable size range (5-30)
    log_likes = np.log1p(likes)  # log1p is log(1+x), safer for 0 values
    size = 5 + (log_likes * 3)  # Base size 5, scale factor 3
    sizes.append(max(5, min(30, size)))  # Clamp between 5 and 30

max_likes = max(favorite_counts) if favorite_counts else 0
min_likes = min(favorite_counts) if favorite_counts else 0
max_size = max(sizes)
min_size = min(sizes)

# Create base colors for clusters
point_colors = [color_map[label] for label in cluster_labels]

# Create data source for bokeh
source = ColumnDataSource(
    data=dict(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        username=df["username"].tolist(),
        emb_text=results["texts"],
        cluster=cluster_labels,
        cluster_prob=cluster_probabilities,
        color=point_colors,
        size=sizes,
        tweet_id=df["tweet_id"].tolist(),
        favorite_count=df["favorite_count"].tolist(),
        retweet_count=df["retweet_count"].tolist(),
    )
)

# Create hover tool
hover = HoverTool(
    tooltips=[
        ("Username", "@username"),
        # ("Tweet ID", "@tweet_id"),
        ("Cluster", "@cluster"),
        # ("Cluster Prob", "@cluster_prob{0.00}"),
        ("Likes", "@favorite_count"),
        # ("Retweets", "@retweet_count"),
        # ("Size", "@size{0.0}"),
        ("Text", "@emb_text{safe}"),
    ]
)

# Create the plot
p = figure(
    width=1000,
    height=800,
    title="Tweet Embeddings (UMAP 10D→2D, HDBSCAN Clustering, Size = Log Likes)",
    tools=["pan", "wheel_zoom", "box_zoom", "reset", "save", hover],
)

# Add scatter plot with variable size based on log likes
scatter = p.scatter(
    "x",
    "y",
    source=source,
    size="size",  # Use the size column for point size
    color="color",
    alpha=0.7,  # Fixed alpha for consistency
    line_color="white",
    line_width=0.5,
)

# Style the plot
p.title.text_font_size = "16pt"
p.xaxis.axis_label = "UMAP Dimension 1"
p.yaxis.axis_label = "UMAP Dimension 2"

# Show the plot
show(p)

print(f"✅ Plotted {len(embeddings_2d)} tweets")
print(f"   - {n_clusters} clusters found")
print(f"   - {n_noise} noise points")
print(f"   - Used 10D UMAP → HDBSCAN → 2D visualization")
print(f"   - Point size based on log likes (min: {min_likes}, max: {max_likes})")
print(f"   - Size range: {min_size:.1f} to {max_size:.1f}")

# %%
# Show like distribution for context
print(f"\n=== LIKE DISTRIBUTION ===")
likes_sorted = sorted(favorite_counts, reverse=True)
print(f"Top 10 most liked tweets: {likes_sorted[:10]}")
print(f"Bottom 10 least liked tweets: {likes_sorted[-10:]}")
print(f"Mean likes: {np.mean(favorite_counts):.1f}")
print(f"Median likes: {np.median(favorite_counts):.1f}")

# Show size distribution/
print(f"\n=== SIZE DISTRIBUTION ===")
sizes_sorted = sorted(sizes, reverse=True)
print(f"Largest sizes: {sizes_sorted[:10]}")
print(f"Smallest sizes: {sizes_sorted[-10:]}")
print(f"Mean size: {np.mean(sizes):.1f}")


# %%
# Enhanced cluster analysis with TF-IDF and ordering by median likes
def clean_text_for_tfidf(text):
    """Clean text for TF-IDF analysis"""
    # Remove usernames, URLs, and clean up
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def get_top_tfidf_words(texts, n_words=10):
    """Get top TF-IDF words for a cluster"""
    if not texts:
        return []

    cleaned_texts = [clean_text_for_tfidf(text) for text in texts]

    # Filter out very short texts
    cleaned_texts = [text for text in cleaned_texts if len(text.split()) > 2]

    if not cleaned_texts:
        return []

    try:

        tfidf = TfidfVectorizer(
            max_features=100,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8,
        )
        tfidf_matrix = tfidf.fit_transform(cleaned_texts)

        # Get average TF-IDF scores
        mean_scores = tfidf_matrix.mean(axis=0).A1
        feature_names = tfidf.get_feature_names_out()

        # Sort by score
        word_scores = list(zip(feature_names, mean_scores))
        word_scores.sort(key=lambda x: x[1], reverse=True)

        return [word for word, score in word_scores[:n_words]]
    except:
        return []


# %%
# add full_text from df to results
results["full_text"] = df["full_text"].tolist()

print("\n=== ENHANCED CLUSTER ANALYSIS ===")

# Calculate cluster statistics for ordering
cluster_stats = {}
for cluster_id in set(cluster_labels):
    cluster_mask = cluster_labels == cluster_id
    cluster_texts = [
        results["full_text"][i]
        for i in range(len(results["full_text"]))
        if cluster_mask[i]
    ]
    cluster_likes = [
        favorite_counts[i] for i in range(len(favorite_counts)) if cluster_mask[i]
    ]
    cluster_sizes = [sizes[i] for i in range(len(sizes)) if cluster_mask[i]]

    # Get indices for this cluster
    cluster_indices = [
        i for i in range(len(cluster_labels)) if cluster_labels[i] == cluster_id
    ]

    cluster_stats[cluster_id] = {
        "texts": cluster_texts,
        "likes": cluster_likes,
        "sizes": cluster_sizes,
        "indices": cluster_indices,
        "median_likes": np.median(cluster_likes) if cluster_likes else 0,
        "mean_likes": np.mean(cluster_likes) if cluster_likes else 0,
        "max_likes": max(cluster_likes) if cluster_likes else 0,
        "count": len(cluster_texts),
    }

# Order clusters by median likes (descending)
ordered_clusters = sorted(
    cluster_stats.items(), key=lambda x: x[1]["median_likes"], reverse=True
)

for cluster_id, stats in ordered_clusters[:10]:
    cluster_name = "NOISE" if cluster_id == -1 else f"CLUSTER {cluster_id}"

    print(f"\n{cluster_name} ({stats['count']} tweets)")
    print(
        f"  Likes - Median: {stats['median_likes']:.1f}, Mean: {stats['mean_likes']:.1f}, Max: {stats['max_likes']}"
    )
    print(
        f"  Size - Mean: {np.mean(stats['sizes']):.1f}, Max: {max(stats['sizes']) if stats['sizes'] else 0:.1f}"
    )

    # Top TF-IDF words
    top_words = get_top_tfidf_words(stats["texts"], n_words=8)
    if top_words:
        print(f"  Top words: {', '.join(top_words)}")
    else:
        print(f"  Top words: [not enough text for analysis]")

    # Top liked tweets from this cluster
    if stats["likes"]:
        # Get top 3 most liked tweets from this cluster
        tweet_like_pairs = list(zip(stats["indices"], stats["likes"], stats["texts"]))
        tweet_like_pairs.sort(key=lambda x: x[1], reverse=True)

        print(f"  Top liked tweets:")
        for i, (idx, likes, text) in enumerate(tweet_like_pairs[:3]):
            username = df.iloc[idx]["username"] if idx < len(df) else "unknown"
            print(
                f"    {i+1}. [{likes} likes] @{username}: {text[:80]}{'...' if len(text) > 80 else ''}"
            )

        if len(tweet_like_pairs) > 3:
            print(f"    ... and {len(tweet_like_pairs) - 3} more tweets")
    else:
        print(f"  No tweets with likes data")

# %%
# Summary statistics
print(f"\n=== SUMMARY STATISTICS ===")
print(f"Total tweets: {len(results['texts'])}")
print(f"Total clusters: {n_clusters}")
print(f"Noise points: {n_noise}")

# Top clusters by median engagement
print(f"\nTop 5 clusters by median likes:")
for i, (cluster_id, stats) in enumerate(ordered_clusters[:5]):
    if cluster_id != -1:  # Skip noise
        cluster_name = f"CLUSTER {cluster_id}"
        print(
            f"  {i+1}. {cluster_name}: {stats['median_likes']:.1f} median likes ({stats['count']} tweets)"
        )

# %%
