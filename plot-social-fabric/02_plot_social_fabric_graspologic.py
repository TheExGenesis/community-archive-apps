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
from bokeh.palettes import Spectral8
import graspologic


DATA_PATH = Path("../data/ca_paper_data.parquet")

# %%
df = pd.read_parquet(DATA_PATH)

# %%
df.head()
"""
tweet_id	account_id	created_at	full_text	retweet_count	favorite_count	reply_to_tweet_id	reply_to_user_id	reply_to_username	archive_upload_id	updated_at	username	temporal_subset	topic
0	1686762764532920321	1.0021918513050583e+18	2023-08-02 15:35:56+00:00	@8chabard it does	0.0	2.0	1.6867615663378924e+18	52333946.0	8chabard	270.0	2025-01-25 18:49:16.732494+00:00	imhinesmi	True	Pharmaceutical Drugs
1	1346709099166121985	1.0021918513050583e+18	2021-01-06 06:44:04+00:00	@moonknower why not	0.0	1.0	1.346600877478568e+18	1.2522775117461832e+18	parafactual	270.0	2025-01-25 18:49:16.732494+00:00	imhinesmi	True	Space exploration
2	1327053481782349827	1.0021918513050583e+18	2020-11-13 00:59:40+00:00	@disconcerta whats the story?	0.0	2.0	1.327052842302124e+18	1.1143560472950784e+18	None	270.0	2025-01-25 18:49:16.732494+00:00	imhinesmi	True	Literature
3	1322302328158957568	1.0021918513050583e+18	2020-10-30 22:20:16+00:00	@grantadever its real fun	0.0	1.0	1.3223013339412234e+18	22525974.0	grantadever	270.0	2025-01-25 18:49:16.732494+00:00	imhinesmi	True	Environmental projects
4
"""
# %%

# %%
month_to_analyze = "2023-11"
# Ensure created_at is properly timezone-aware
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
# Drop any rows where date parsing failed
df = df.dropna(subset=["created_at"])

# %%
# Use direct datetime comparison instead of string formatting (much faster)
fourth_week_start = pd.Timestamp(month_to_analyze + "-22", tz="UTC")
fourth_week_end = pd.Timestamp(month_to_analyze + "-28", tz="UTC")


# Filter directly to the week we want (avoiding the slow month filter)
monthly_df = df[
    (df["created_at"] >= fourth_week_start) & (df["created_at"] <= fourth_week_end)
]
# %%

# let's release df from memory
del df
# %%
# We are interested in replies, so we look at 'username' and 'reply_to_username'
interactions_df = monthly_df.dropna(subset=["username", "reply_to_username"])
# Remove self-loops
interactions_df = interactions_df[
    interactions_df["username"] != interactions_df["reply_to_username"]
]
G = nx.MultiDiGraph()
for _, row in interactions_df.iterrows():
    G.add_edge(row["username"], row["reply_to_username"], timestamp=row["created_at"])
ego_node = "exgenesis"
ego_graph = nx.ego_graph(G, ego_node, radius=1)
# %%
ego_graph_undirected = ego_graph.to_undirected()

# %%
import graspologic
import numpy as np
from graspologic.plot import heatmap
from graspologic.simulations import sbm
from graspologic.embed import AdjacencySpectralEmbed

# %%
adj = nx.to_numpy_array(ego_graph_undirected)
adj_sym = (adj + adj.T) / 2  # symmetrise *before* any scaling
ase = AdjacencySpectralEmbed(n_components=2)
X = ase.fit_transform(adj_sym)  # single array
# %%
from graspologic.plot import pairplot

plot = pairplot(X)

# %%
from graspologic.plot.plot import networkplot

x_pos = X[:, 0]
y_pos = X[:, 1]
degrees = np.sum(adj_sym, axis=0)
node_ids = list(ego_graph_undirected.nodes())

plot = networkplot(
    adjacency=adj_sym,
    x=x_pos,
    y=y_pos,
    node_hue=None,
    palette="deep",
    node_size=degrees,
    node_sizes=(20, 200),
    edge_hue="source",
    edge_alpha=0.5,
    edge_linewidth=0.5,
    labels=node_ids,
)

# %%
