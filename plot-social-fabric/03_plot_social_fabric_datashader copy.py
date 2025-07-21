# %%
"""
Social Network Visualization with Improved Label Legibility

This script creates interactive social network visualizations using NetworkX,
Holoviews, and Datashader with configurable label styling options.

"""

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


# %%
DATA_PATH = Path("../data/tweets_2023_11_week4.parquet")

# Define variables used in plot titles
ego_node = "network"  # Default name for network analysis
fourth_week_start = pd.Timestamp("2023-11-22", tz="UTC")
fourth_week_end = pd.Timestamp("2023-11-28", tz="UTC")


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
# let's see what ppl were talking about at the time
pd.set_option("display.max_colwidth", None)
df = df.set_index("tweet_id", inplace=False, drop=False)
df.sort_values("favorite_count", ascending=False).head(50)[
    [
        "full_text",
        "username",
        "reply_to_username",
        "favorite_count",
        "retweet_count",
        "created_at",
    ]
]

# %%
# let's add a column "num_replies", that tells us how many replies a tweet has in the df, count occurrences of reply_to_tweet_id
# dict from tweet_id -> int
from collections import Counter

reply_to_tweet_id_counts = Counter(df["reply_to_tweet_id"].values)
df["num_replies"] = df["tweet_id"].map(reply_to_tweet_id_counts)

# %%
# let's see what the tweets with most engagement were and who wrote them
df.sort_values("num_replies", ascending=False).head(50)[
    [
        "full_text",
        "username",
        "reply_to_username",
        "favorite_count",
        "retweet_count",
        "created_at",
        "num_replies",
    ]
]

# %%
# Let's see which accounts were most engaged with
interactions_df = df.dropna(subset=["username", "reply_to_username"])
# Remove self-loops
interactions_df = interactions_df[
    interactions_df["username"] != interactions_df["reply_to_username"]
]
interactions_df = interactions_df[
    interactions_df["reply_to_username"].isin(interactions_df["username"])
]

# now let's count how many total replies and favorites and retweets and tweets each account has
reply_to_tweet_id_counts = Counter(
    interactions_df["reply_to_username"].values
)  # dict from username -> int
total_replies_made = interactions_df.groupby("username")["reply_to_username"].count()
total_replies_received = total_replies_made.index.map(reply_to_tweet_id_counts)
total_favorites = interactions_df.groupby("username")["favorite_count"].sum()
total_retweets = interactions_df.groupby("username")["retweet_count"].sum()
total_tweets = interactions_df.groupby("username")["tweet_id"].count()

user_df = pd.DataFrame(
    {
        "username": total_replies_made.index,
        "total_replies_made": total_replies_made.values,
        "total_replies_received": total_replies_received.values,
        "total_favorites": total_favorites.values,
        "total_retweets": total_retweets.values,
        "total_tweets": total_tweets.values,
    }
)
user_df.sort_values("total_replies_received", ascending=False).head(50)


# %%

edge_mode = "multiple"
# Build graph according to edge_mode
if edge_mode == "multiple":
    G = nx.MultiDiGraph()
    for _, row in interactions_df.iterrows():
        G.add_edge(
            row["username"], row["reply_to_username"], timestamp=row["created_at"]
        )
elif edge_mode == "weighted":
    G = nx.DiGraph()
    # Count interactions between each pair
    edge_counts = (
        interactions_df.groupby(["username", "reply_to_username"])
        .size()
        .reset_index(name="weight")
    )
    for _, row in edge_counts.iterrows():
        G.add_edge(row["username"], row["reply_to_username"], weight=row["weight"])
elif edge_mode == "binary":
    G = nx.DiGraph()
    # Just add one edge per unique pair
    unique_pairs = interactions_df[["username", "reply_to_username"]].drop_duplicates()
    for _, row in unique_pairs.iterrows():
        G.add_edge(row["username"], row["reply_to_username"])
else:
    raise ValueError(
        f"Invalid edge_mode: {edge_mode}. Must be 'multiple', 'weighted', or 'binary'."
    )

print(f"Graph built with edge_mode='{edge_mode}'")
print(f"Total interactions in data: {len(interactions_df)}")
print(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")


# %%
# Validation: Print each account's top 5 interacted with accounts
print("\n" + "=" * 80)
print("INTERACTION VALIDATION - Top 5 accounts each user interacted with:")
print("=" * 80)

# Get interaction counts for the original directed graph
if edge_mode == "weighted":
    # For weighted edges, we already have the weights
    interaction_counts = {}
    for u, v, data in G.edges(data=True):
        if u not in interaction_counts:
            interaction_counts[u] = {}
        if v not in interaction_counts:
            interaction_counts[v] = {}

        weight = data.get("weight", 1)
        interaction_counts[u][v] = weight
        # For undirected analysis, also count reverse
        interaction_counts[v][u] = weight

elif edge_mode == "binary":
    # For binary edges, count from the original interactions_df
    interaction_counts = {}
    for _, row in interactions_df.iterrows():
        u, v = row["username"], row["reply_to_username"]
        if u not in interaction_counts:
            interaction_counts[u] = {}
        if v not in interaction_counts:
            interaction_counts[v] = {}

        interaction_counts[u][v] = interaction_counts[u].get(v, 0) + 1
        interaction_counts[v][u] = interaction_counts[v].get(u, 0) + 1

else:  # multiple
    # For multiple edges, count from the original interactions_df
    interaction_counts = {}
    for _, row in interactions_df.iterrows():
        u, v = row["username"], row["reply_to_username"]
        if u not in interaction_counts:
            interaction_counts[u] = {}
        if v not in interaction_counts:
            interaction_counts[v] = {}

        interaction_counts[u][v] = interaction_counts[u].get(v, 0) + 1
        interaction_counts[v][u] = interaction_counts[v].get(u, 0) + 1

# Print top 5 for each account, sorted by degree (most connected first)
node_degrees = dict(G.degree())
sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)

for node, degree in sorted_nodes[:20]:  # Show top 20 most connected accounts
    if node in interaction_counts:
        top_interactions = sorted(
            interaction_counts[node].items(), key=lambda x: x[1], reverse=True
        )[:5]

        print(f"\n@{node} (degree: {degree}):")
        for partner, count in top_interactions:
            print(f"  → @{partner}: {count} interactions")
    else:
        print(f"\n@{node} (degree: {degree}): No interaction data found")


# %%
G_undirected = nx.Graph(G.to_undirected())  # Convert to simple graph

# Filter out nodes with degree < 3, to make the graph less busy
nodes_to_remove = [node for node, degree in G_undirected.degree() if degree < 3]
G_undirected.remove_nodes_from(nodes_to_remove)
print(f"Removed {len(nodes_to_remove)} nodes with degree < 3")
print(f"Remaining nodes: {G_undirected.number_of_nodes()}")

# %%
import graspologic
import numpy as np
from graspologic.plot import heatmap
from graspologic.simulations import sbm
from graspologic.embed import AdjacencySpectralEmbed
from graspologic.embed import LaplacianSpectralEmbed
from graspologic.plot import pairplot


adj = nx.to_numpy_array(G_undirected)
adj_sym = (adj + adj.T) / 2  # symmetrise *before* any scaling
ase = AdjacencySpectralEmbed(n_components=2)
log_adj_sym = np.log(adj_sym + 1)
X = ase.fit_transform(log_adj_sym)  # single array

# %%
# heatmap
heatmap(adj, title="Adjacency Matrix")

# %%
from bokeh.plotting import output_notebook, show, figure, ColumnDataSource
from bokeh.models import HoverTool, LabelSet


def plot_graph_embedding_bokeh(
    G, X, title="Interactive ASE Embedding (raw) - Hover to see node names"
):
    """
    G: networkx.Graph
    X: np.ndarray, shape (n_nodes, 2)
    """
    output_notebook()
    node_names = list(G.nodes())
    node_degrees = [G.degree(node) for node in node_names]
    node_sizes = [max(5, min(d * 2, 30)) for d in node_degrees]
    source = ColumnDataSource(
        data=dict(
            x=X[:, 0],
            y=X[:, 1],
            names=node_names,
            degrees=node_degrees,
            sizes=node_sizes,
        )
    )
    p = figure(
        width=800,
        height=600,
        title=title,
        tools="pan,wheel_zoom,box_zoom,reset,hover,save",
    )
    hover = p.select_one(HoverTool)
    hover.tooltips = [
        ("Node", "@names"),
        ("Degree", "@degrees"),
        ("X", "@x{0.000}"),
        ("Y", "@y{0.000}"),
    ]
    p.circle(
        "x",
        "y",
        source=source,
        size="sizes",
        color="darkblue",
        alpha=0.7,
        line_color="white",
        line_width=1,
    )
    labels = LabelSet(
        x="x",
        y="y",
        text="names",
        x_offset=8,
        y_offset=8,
        source=source,
        text_font_size="9pt",
        text_color="black",
        background_fill_color="white",
        background_fill_alpha=0.8,
    )
    p.add_layout(labels)
    show(p)


# %%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def graph_stats_df(G, X):
    # G: networkx.Graph, X: np.ndarray (n_nodes, 2)
    stats = {
        "degree": pd.Series(dict(G.degree())),
        "clustering": pd.Series(nx.clustering(G)),
        "closeness": pd.Series(nx.closeness_centrality(G)),
        "eigenvector": pd.Series(nx.eigenvector_centrality(G)),
        "pagerank": pd.Series(nx.pagerank(G)),
    }
    df = pd.DataFrame(stats)
    df["log_degree"] = np.log1p(df["degree"])
    df["x_coord"] = X[:, 0]
    df["y_coord"] = X[:, 1]
    return df


def print_embedding_corr_table(G, X):
    stats_to_plot = ["degree", "clustering", "closeness", "eigenvector", "pagerank"]
    df = graph_stats_df(G, X)
    corr_results = [
        {
            "statistic": stat,
            "x_pearson": pearsonr(df["x_coord"], df[stat])[0],
            "x_p_value": pearsonr(df["x_coord"], df[stat])[1],
            "y_pearson": pearsonr(df["y_coord"], df[stat])[0],
            "y_p_value": pearsonr(df["y_coord"], df[stat])[1],
            "x_spearman": spearmanr(df["x_coord"], df[stat])[0],
            "y_spearman": spearmanr(df["y_coord"], df[stat])[0],
        }
        for stat in stats_to_plot
    ]
    corr_df = pd.DataFrame(corr_results)
    print("Correlation Analysis: Embedding Dimensions vs Graph Statistics")
    print("=" * 70)
    print(corr_df.round(3))


def plot_embedding_corr_heatmap(G, X):
    stats_to_plot = ["degree", "clustering", "closeness", "eigenvector", "pagerank"]
    df = graph_stats_df(G, X)
    cols = ["x_coord", "y_coord"] + stats_to_plot
    corr_matrix = df[cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) | np.eye(
        corr_matrix.shape[0], dtype=bool
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar=True,
    )
    plt.title(
        "Correlation Matrix: Embedding Coordinates vs Graph Statistics (Lower Triangle, No Diagonal)"
    )
    plt.tight_layout()
    plt.show()


stats_to_plot = ["degree", "clustering", "closeness", "eigenvector", "pagerank"]


# %%
# Non-log transformed ASE
ase_raw = AdjacencySpectralEmbed(n_components=2)
X_raw = ase_raw.fit_transform(adj_sym)
plot_raw = pairplot(X_raw, title="Adjacency Spectral Embedding (raw)")
plot_raw.fig.suptitle("Adjacency Spectral Embedding (raw)", fontsize=16)


# %%
plot_graph_embedding_bokeh(G_undirected, X_raw)


# %%
# the x-axis seems to be the degree of the node in the graph but holy shit it looks like the y-axis is something like how "meditation-y" an account is (up) vs how "techy" (down) - it's not perfect especially bc it's only 1 week of tweets but it's my best hypothesis so far

# %%
plot_embedding_corr_heatmap(G_undirected, X_raw)


# %%
# # let's try the pass-to-rank trick
from scipy.stats import rankdata
from graspologic.embed import AdjacencySpectralEmbed

# 1. build & symmetrise adjacency
adj = nx.to_numpy_array(G_undirected)
adj_sym = (adj + adj.T) / 2

# 2. pass-to-ranks (global ranks, then [0,1])
r = rankdata(adj_sym.ravel(), method="ordinal").astype(float)
ptr = r.reshape(adj_sym.shape)
ptr /= ptr.max()
ptr = (ptr + ptr.T) / 2  # enforce perfect symmetry

# 3. spectral embedding
ase = AdjacencySpectralEmbed(n_components=2)
X_ranked = ase.fit_transform(ptr)

plot_raw = pairplot(X_ranked, title="Adjacency Spectral Embedding (pass-to-ranks)")


# %%
plot_graph_embedding_bokeh(G_undirected, X_ranked)

# %%
# I can't tell if the X azxis axis has any meaning.... but the shape of the blob seems more reasonable...

# %%
plot_embedding_corr_heatmap(G_undirected, X_ranked)


# %%
lsae = LaplacianSpectralEmbed(n_components=2)
X_lsae = lsae.fit_transform(adj_sym)


# %%
plot = pairplot(X_lsae, title="Laplacian Spectral Embedding (2 components)")
plot.fig.suptitle("Laplacian Spectral Embedding (2 components)", fontsize=16)

# %%
plot_graph_embedding_bokeh(G_undirected, X_lsae)


# %%
# looks like non laplacian ASE but with 2 weird outliers

# %%
import matplotlib.pyplot as plt

# plot degree distribution
degrees = pd.Series(dict(G_undirected.degree())).sort_values(ascending=False)


# %%
lsa = LaplacianSpectralEmbed(form="R-DAD", n_components=2)
X_lsa_rdad = lsa.fit_transform(adj_sym)

# %%
plot = pairplot(X_lsa_rdad, title="Laplacian Spectral Embedding (R-DAD form)")
plot.fig.suptitle("Laplacian Spectral Embedding (R-DAD form)", fontsize=16)

# %%
#  can't tell much of a difference

# %%
from sklearn.metrics import pairwise_distances

A = nx.to_numpy_array(G_undirected)
sim = 1 - pairwise_distances(A, metric="jaccard")
X_jaccard_sim = AdjacencySpectralEmbed(n_components=2).fit_transform(sim)
plot = pairplot(X, title="Adjacency Spectral Embedding (Jaccard similarity)")
plot.fig.suptitle("Adjacency Spectral Embedding (Jaccard similarity)", fontsize=16)

# %%
plot_graph_embedding_bokeh(G_undirected, X_jaccard_sim)


# %%
# the others look the same but this one kinda looks more like a 3 pointed simplex, are there 3 distinguishable coordinates?

# %%
plot_embedding_corr_heatmap(G_undirected, X_jaccard_sim)

# %%
# fascinating! so jaccard similarity actually gives us closeness centrality and clustering coefficient / degree, AND a third thing? I can't tell what tho

# okay so the diagonal y=-x diagonal is the degree - and then a component of X that correlates with closeness centrality, or how ~ close a user is to everyone else; and a component of y that correlates w clustering coefficient or how cliquey a node's neighborhood is. I don't know the small accounts at the extremes to judge well enough

# %%
# I wonder if the pass-to-rank trick helps us here

# %%
# --- Node2Vec embedding with UMAP 2D projection ---------------------------
import umap

node2vec = Node2Vec(
    G_undirected,
    dimensions=64,
    walk_length=20,
    num_walks=200,
    workers=4,
    p=1,
    q=1,
    seed=42,
)
n2v_model = node2vec.fit(window=10, min_count=1, epochs=1)
nodes = list(G_undirected.nodes())
X_n2v = np.array([n2v_model.wv[str(node)] for node in nodes])

# %%
# refresh umap import in jupyter cell

X_n2v_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(X_n2v)
plot = pairplot(X_n2v_2d, title="Node2Vec Embedding (UMAP to 2D)")
plot.fig.suptitle("Node2Vec Embedding (UMAP to 2D)", fontsize=16)


# %%
from graspologic.plot.plot import networkplot
import networkx.algorithms.community as nx_comm

# Use Louvain community detection on the graph structure
communities = nx_comm.louvain_communities(G_undirected)
n_clusters = len(communities)

# Create cluster labels mapping
cluster_labels = np.zeros(G_undirected.number_of_nodes(), dtype=int)
node_list = list(G_undirected.nodes())
for cluster_id, community in enumerate(communities):
    for node in community:
        node_idx = node_list.index(node)
        cluster_labels[node_idx] = cluster_id

x_pos = X_n2v_2d[:, 0]
y_pos = X_n2v_2d[:, 1]
degrees = np.sum(adj_sym, axis=0)
node_ids = list(G_undirected.nodes())

plot = networkplot(
    adjacency=adj_sym,
    x=x_pos,
    y=y_pos,
    node_hue=cluster_labels,
    palette="deep",
    node_size=degrees,
    node_sizes=(20, 200),
    edge_hue="source",
    edge_alpha=0.5,
    edge_linewidth=0.5,
    title="Node2Vec Embedding Network Plot (Louvain Communities)",
)


# %%
import networkx as nx
import holoviews as hv
from holoviews import opts
import holoviews.operation.datashader as hd

hv.extension("bokeh")


# %%


# %%
def get_layout(graph, layout_type, X=None):
    if layout_type == "node2vec" and X is not None:
        node_list = list(graph.nodes())
        return {node: (X[i, 0], X[i, 1]) for i, node in enumerate(node_list)}
    elif layout_type == "force_directed":
        return nx.layout.fruchterman_reingold_layout(graph, seed=42)
    else:
        raise ValueError(f"Unknown layout_type: {layout_type}")


def plot_layout(graph, layout_type, X=None, cluster_labels=None, title=""):
    layout = get_layout(graph, layout_type, X)
    node_list = list(graph.nodes())
    nodes_df = pd.DataFrame(
        {
            "index": node_list,
            "x": [layout[n][0] for n in node_list],
            "y": [layout[n][1] for n in node_list],
            "cluster": cluster_labels if cluster_labels is not None else 0,
        }
    )
    edges_df = pd.DataFrame([{"source": u, "target": v} for u, v in graph.edges()])
    nodes_elem = hv.Nodes(nodes_df, ["x", "y", "index"], ["cluster"])
    graph_elem = hv.Graph((edges_df, nodes_elem))
    node_degrees = dict(graph.degree())
    top_percentile = np.percentile(list(node_degrees.values()), 80)
    labels_df = pd.DataFrame(
        [
            {"x": layout[node][0], "y": layout[node][1], "index": node}
            for node in node_list
            if node_degrees[node] >= top_percentile
        ]
    )
    labels_elem = (
        hv.Labels(labels_df, ["x", "y"], "index") if not labels_df.empty else None
    )
    bundled = hd.bundle_graph(graph_elem)
    shaded = hd.datashade(bundled, width=800, height=800, cmap="lightgray")
    plot = shaded * graph_elem
    if labels_elem is not None:
        plot = plot * labels_elem
    styled_plot = plot.opts(
        opts.Graph(edge_alpha=0),
        opts.Nodes(size=20, color="cluster", cmap="viridis", colorbar=True),
        opts.Labels(text_font_size="10pt", text_color="black"),
        opts.Overlay(title=title, width=800, height=800),
    )
    hv.output(styled_plot)
    return styled_plot


import holoviews as hv

hv.extension("bokeh")  # registers the backend with jupyter_bokeh
force_directed_plot = plot_layout(
    G_undirected,
    "force_directed",
    cluster_labels=cluster_labels,
    title=f"Force-Directed Layout: @{ego_node} - Louvain Communities",
)

# %%
import holoviews as hv

hv.extension("bokeh")  # registers the backend with jupyter_bokeh

# %%
# Show cluster info
cluster_info = pd.DataFrame(
    {"node": list(G_undirected.nodes()), "cluster": cluster_labels}
)
cluster_counts = cluster_info.groupby("cluster").size().sort_values(ascending=False)
print(f"Louvain found {len(set(cluster_labels))} communities:")
print(cluster_counts.head(10))


# %%
# Create plots


node2vec_plot = plot_layout(
    G_undirected,
    "node2vec",
    X=X_n2v_2d,
    cluster_labels=cluster_labels,
    title=f"Node2Vec Layout: @{ego_node} - Louvain Communities",
)


# %%
force_directed_plot


# %%
node2vec_plot


# %%
