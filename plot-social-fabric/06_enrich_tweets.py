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


def save_results(results: dict, filepath: str, format: str = "json") -> None:
    """Save results to file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj

        with open(filepath.with_suffix(".json"), "w") as f:
            json.dump(convert_numpy(results), f, indent=2)
    elif format == "npy":
        # For numpy arrays, save as .npy file
        if isinstance(results, np.ndarray):
            np.save(filepath.with_suffix(".npy"), results)
        else:
            # If it's a dict with embeddings, save embeddings as .npy and metadata as .json
            if "embeddings" in results:
                embeddings = np.array(results["embeddings"])
                np.save(filepath.with_suffix(".npy"), embeddings)

                # Save metadata without embeddings
                metadata = {k: v for k, v in results.items() if k != "embeddings"}
                with open(filepath.with_suffix("_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
            else:
                # Fallback to pickle for complex objects
                import pickle

                with open(filepath.with_suffix(".pkl"), "wb") as f:
                    pickle.dump(results, f)
    else:
        import pickle

        with open(filepath.with_suffix(".pkl"), "wb") as f:
            pickle.dump(results, f)


class TweetData(TypedDict):
    tweet_id: str
    account_id: str
    created_at: datetime
    full_text: str
    retweet_count: int
    favorite_count: int
    reply_to_tweet_id: Optional[str]
    reply_to_user_id: Optional[str]
    reply_to_username: Optional[str]
    quoted_tweet_id: Optional[str]
    username: str


class ConversationTree(TypedDict):
    root: str
    children: Dict[str, List[str]]
    parents: Dict[str, str]
    paths: Dict[str, List[str]]


def clean_tweet_text(text: str, max_length: int = 32000) -> str:
    text = re.sub(r"This Post is from a suspended account.*", "", text)
    text = re.sub(r"https://t\.co/\w+", "", text)
    text = re.sub(r"^RT @[A-Za-z0-9_]+: ", "", text)
    text = re.sub(r"^(\s*@\w+\s*)+", "", text)
    text = text.strip()
    return text[:max_length] + "..." if len(text) > max_length else text


def extract_quoted_text(
    tweet: TweetData, tweets_dict: Dict[str, TweetData]
) -> Optional[str]:
    """Extract quoted text using quoted_tweet_id if available."""
    quoted_id = tweet.get("quoted_tweet_id")
    if quoted_id and pd.notna(quoted_id) and quoted_id in tweets_dict:
        quoted_tweet = tweets_dict[quoted_id]
        return clean_tweet_text(quoted_tweet["full_text"])
    return None


def build_conversation_trees(tweets: List[TweetData]) -> Dict[str, ConversationTree]:
    trees: Dict[str, ConversationTree] = {}
    tweets_dict = {t["tweet_id"]: t for t in tweets}

    parents = {}
    children = defaultdict(list)

    for tweet in tweets:
        reply_to = tweet.get("reply_to_tweet_id")
        if reply_to and reply_to in tweets_dict:
            parents[tweet["tweet_id"]] = reply_to
            children[reply_to].append(tweet["tweet_id"])

    roots = [tid for tid in tweets_dict if tid not in parents]

    for root_id in roots:
        tree: ConversationTree = {
            "root": root_id,
            "children": defaultdict(list),
            "parents": {},
            "paths": {},
        }

        queue = deque([(root_id, [root_id])])
        visited = set()
        while queue:
            current_id, path = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)
            current_children = children.get(current_id, [])
            if not current_children:
                tree["paths"][current_id] = path
            for child_id in current_children:
                tree["parents"][child_id] = current_id
                tree["children"][current_id].append(child_id)
                queue.append((child_id, path + [child_id]))

        trees[root_id] = tree

    return trees


def get_thread_embedding_text(
    tweet_id: str,
    tree: ConversationTree,
    tweets_dict: Dict[str, TweetData],
    max_chars: int = 1024,
) -> str:
    parts = []

    # Build user alias mapping
    user_aliases = {}
    alias_counter = 0

    def get_user_alias(user_id: str) -> str:
        nonlocal alias_counter
        if user_id not in user_aliases:
            user_aliases[user_id] = f"user{chr(ord('A') + alias_counter)}"
            alias_counter += 1
        return user_aliases[user_id]

    def format_tweet(tid: str) -> str:
        tweet = tweets_dict[tid]
        user_alias = get_user_alias(tweet["account_id"])
        text = clean_tweet_text(tweet["full_text"])
        quoted = extract_quoted_text(tweet, tweets_dict)
        if quoted:
            text = f"{text}\n[quoted] {quoted}"
        return f"{user_alias}: {text}"

    root_id = tree["root"]
    if root_id != tweet_id:
        parts.append(format_tweet(root_id))

    current_id = tree["parents"].get(tweet_id)
    context_parts = []
    while current_id and current_id != root_id:
        context_parts.append(format_tweet(current_id))
        current_id = tree["parents"].get(current_id)
    parts.extend(reversed(context_parts))

    parts.append(format_tweet(tweet_id))

    # Apply priority-based selection similar to get_tweets.py
    if len(parts) == 1:
        check_order = [parts[0]]
    elif len(parts) == 2:
        check_order = [parts[1], parts[0]]  # current, then root
    elif len(parts) == 3:
        check_order = [parts[2], parts[1], parts[0]]  # current, context, root
    else:
        # current, previous, root, then others in reverse order
        check_order = [parts[-1], parts[-2], parts[0]]
        check_order.extend(reversed(parts[1:-2]))

    # Select parts without exceeding max_chars
    selected_set = set()
    chars = 0
    for part in check_order:
        part_len = len(part) + 1
        if chars + part_len > max_chars:
            break
        selected_set.add(part)
        chars += part_len

    # Assemble selected parts in original order, adding ellipsis for gaps
    selected = []
    for part in parts:
        if part in selected_set:
            selected.append(part)
        else:
            if "<...>" not in selected:
                selected.append("<...>")

    return (
        "\n".join(selected)
        if selected
        else clean_tweet_text(tweets_dict[tweet_id]["full_text"])
    )


# %%
DATA_PATH = Path("../data/tweets_2023_11_week4.parquet")

# %%
df = pd.read_parquet(DATA_PATH)

# %%
df.head()
"""
	tweet_id	account_id	username	account_display_name	created_at	full_text	retweet_count	favorite_count	reply_to_tweet_id	reply_to_user_id	reply_to_username	quoted_tweet_id	conversation_id
0	1728926591877808515	881192533056827392	goblinodds	goblin	2023-11-26T23:59:56+00:00	@mechanical_monk --thinking "oh no what a wast...	0	4	1728926283818762524	881192533056827392	goblinodds	None	1728859285013307870
1	1728926513314361364	322603863	exgenesis	❤️‍🔥 xiq	2023-11-26T23:59:37+00:00	oh man just finished the hunter x hunter anime...	0	37	None	None	None	None	1728926513314361364
2	1728926513863766226	282948199	algekalipso	Captain Pleasure, Andrés Gómez Emilsson	2023-11-26T23:59:37+00:00	It's not for everyone. I think that if one is ...	0	4	1728925172642746465	204967813	samuelxgarcia	None	None

"""
# %%

# Build trees from df
monthly_tweets = df.to_dict("records")
tweets_dict = {t["tweet_id"]: t for t in monthly_tweets}
trees = build_conversation_trees(monthly_tweets)

# Create tweet_to_tree mapping
tweet_to_tree = {
    tid: tree_key
    for tree_key, tree in trees.items()
    for tid in tree["children"] or [tree_key]
}


# Add emb_text
def create_emb_text(row, tweets_dict, trees, tweet_to_tree):
    tree_key = tweet_to_tree.get(row["tweet_id"])
    if tree_key:
        return get_thread_embedding_text(row["tweet_id"], trees[tree_key], tweets_dict)
    # Top-level tweet - add userA prefix
    text = clean_tweet_text(row["full_text"])
    quoted = extract_quoted_text(row, tweets_dict)
    if quoted:
        text = f"{text}\n[quoted] {quoted}"
    return f"userA: {text}"


df["emb_text"] = df.apply(
    lambda row: create_emb_text(row, tweets_dict, trees, tweet_to_tree), axis=1
)
# %%
print(df[["tweet_id", "emb_text"]].head(50))
# %%
for i in range(10):
    print(df.loc[i]["tweet_id"])
    print(df.loc[i]["emb_text"])
    print("-" * 100)
# %%
# print text of longest few emb_text
# print full length of string

longest = (
    df["emb_text"].apply(lambda x: (len(x), x)).sort_values(ascending=False).head(50)
)
for i in range(50):
    print("-" * 100)
    print(longest.index[i])
    print(longest.iloc[i][1])


# %%

texts = df["emb_text"].tolist()

# Get individual Modal functions
embed_texts_only = modal.Function.from_name(
    "embedding-clustering-service", "embed_texts_only"
)
reduce_dimensions = modal.Function.from_name(
    "embedding-clustering-service", "reduce_dimensions"
)
cluster_embeddings = modal.Function.from_name(
    "embedding-clustering-service", "cluster_embeddings"
)

print(f"Processing {len(texts)} texts...")

# Phase 1: Embed in batches using dedicated embedding function
BATCH_SIZE = 2000
batched_texts = list(partition_all(BATCH_SIZE, texts))
print(f"Embedding in {len(batched_texts)} batches of ~{BATCH_SIZE} texts each...")

# Embed all batches in parallel - much more efficient!
print("Submitting all batches...")

batch_calls = []
for batch in tqdm(batched_texts, desc="Submitting batches"):
    call = embed_texts_only.spawn(batch)
    batch_calls.append(call)

print("Collecting results...")
embedding_batches = []
for call in tqdm(batch_calls, desc="Collecting embeddings"):
    embeddings = call.get()
    embedding_batches.append(embeddings)

# Combine all embeddings
embeddings = []
for batch in embedding_batches:
    embeddings.extend(batch)

print(f"Got {len(embeddings)} embeddings")

# Save embeddings locally
embeddings_data = {
    "embeddings": embeddings,
    "texts": texts,
    "n_texts": len(texts),
    "timestamp": datetime.now().isoformat(),
}
save_results(embeddings_data, "tweet_embeddings", "npy")
print("✅ Embeddings saved to tweet_embeddings.npy + tweet_embeddings_metadata.json")

# Phase 2: Dimension reduction to 2D
print("Reducing dimensions to 2D...")
embeddings_2d, reduction_time = reduce_dimensions.remote(
    np.array(embeddings),
    n_components=2,
    n_neighbors=min(15, len(texts) - 1),
    min_dist=0.1,
)

# Phase 3: Clustering
print("Clustering...")
# Auto-estimate clustering parameters
n_texts = len(texts)
if n_texts < 50:
    min_cluster_size = max(2, int(n_texts * 0.05))
    min_samples = max(1, int(n_texts * 0.02))
else:
    min_cluster_size = max(5, int(n_texts * 0.01))
    min_samples = max(2, int(n_texts * 0.005))

cluster_results = cluster_embeddings.remote(
    embeddings_2d, min_cluster_size=min_cluster_size, min_samples=min_samples
)

# Combine all results
final_results = {
    "embeddings": embeddings,
    "embeddings_2d": embeddings_2d.tolist(),
    "clustering": cluster_results,
    "texts": texts,
    "n_texts": len(texts),
    "reduction_time": reduction_time,
    "parameters_used": {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "n_components": 2,
        "batch_size": BATCH_SIZE,
    },
    "timestamp": datetime.now().isoformat(),
}

# Save final results locally - embeddings as .npy, rest as .json
save_results(final_results, "embedded_clustered_tweets", "npy")
print(
    f"✅ Complete results saved to embedded_clustered_tweets.npy + embedded_clustered_tweets_metadata.json"
)
print(f"   - {len(texts)} texts processed")
print(f"   - {cluster_results['n_clusters']} clusters found")
print(f"   - {cluster_results['n_noise']} noise points")

# %%

# load embeddings from embedded_clustered_tweets

results = load_results("embedded_clustered_tweets")
print(f"Loaded {len(results['texts'])} texts")
print(f"Embeddings shape: {np.array(results['embeddings']).shape}")
print(f"Found {results['clustering']['n_clusters']} clusters")
# %%
# Create bokeh plot data
embeddings = np.array(results["embeddings"])


# %%
# plot embeddings_2d


# Create color mapping for clusters
unique_labels = list(set(cluster_labels))
colors = [f"#{i:06x}" for i in range(0, 0xFFFFFF, 0xFFFFFF // len(unique_labels))]
color_map = dict(zip(unique_labels, colors))
point_colors = [color_map[label] for label in cluster_labels]

# Create data source for bokeh
source = ColumnDataSource(
    data=dict(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        username=df["username"].tolist(),
        emb_text=df["emb_text"].tolist(),
        cluster=cluster_labels,
        color=point_colors,
        tweet_id=df["tweet_id"].tolist(),
        favorite_count=df["favorite_count"].tolist(),
        retweet_count=df["retweet_count"].tolist(),
    )
)

# Create hover tool
hover = HoverTool(
    tooltips=[
        ("Username", "@username"),
        ("Tweet ID", "@tweet_id"),
        ("Cluster", "@cluster"),
        ("Likes", "@favorite_count"),
        ("Retweets", "@retweet_count"),
        ("Text", "@emb_text{safe}"),
    ]
)

# Create the plot
p = figure(
    width=800,
    height=600,
    title="Tweet Embeddings (2D UMAP)",
    tools=["pan", "wheel_zoom", "box_zoom", "reset", "save", hover],
)

# Add scatter plot
scatter = p.scatter(
    "x",
    "y",
    source=source,
    size=8,
    color="color",
    alpha=0.7,
    line_color="white",
    line_width=0.5,
)

# Style the plot
p.title.text_font_size = "16pt"
p.xaxis.axis_label = "UMAP Dimension 1"
p.yaxis.axis_label = "UMAP Dimension 2"

# Show the plot
show(p)

print(f"Plotted {len(embeddings_2d)} tweets with {len(unique_labels)} clusters")

# %%
