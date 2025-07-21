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
# Your data appears to be from the past, so I'm using Nov 2020 as an example.
# You can change the start and end dates to any month you are interested in.
month_to_analyze = "2024-11"
# %%
# Handle invalid dates by coercing errors to NaT
df["created_at"] = pd.to_datetime(df["created_at"], format="mixed", errors="coerce")
# Drop any rows where date parsing failed
df = df.dropna(subset=["created_at"])
monthly_df = df[df.created_at.dt.strftime("%Y-%m") == month_to_analyze]

# We are interested in replies, so we look at 'username' and 'reply_to_username'
interactions_df = monthly_df.dropna(subset=["username", "reply_to_username"])
# Create a directed multigraph to allow multiple edges
G = nx.MultiDiGraph()
# %%
# Add edges with timestamps
for _, row in interactions_df.iterrows():
    G.add_edge(row["username"], row["reply_to_username"], timestamp=row["created_at"])
# %%
# Get all edges and count frequencies
edge_counts = {}
for u, v in G.edges():
    pair = tuple(sorted([u, v]))
    edge_counts[pair] = edge_counts.get(pair, 0) + 1

# Sort by frequency and get top pairs
top_pairs = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)[:10]

print(f"\nMost frequent interactions in {month_to_analyze}:")
for (user1, user2), count in top_pairs:
    print(f"{user1} <-> {user2}: {count} interactions")
# %%

# Plot degree distribution with Bokeh
degrees = [d for n, d in G.degree() if d > 0]
if degrees:
    hist, edges = np.histogram(degrees, bins=50)

    p = figure(
        title=f"Degree Distribution in {month_to_analyze}",
        x_axis_label="Degree",
        y_axis_label="Count",
        x_axis_type="log",
        y_axis_type="log",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    p.quad(
        top=hist,
        bottom=0.1,  # Start bottom at 0.1 to avoid log(0) issues
        left=edges[:-1],
        right=edges[1:],
        fill_color=Spectral8[0],
        line_color="white",
        alpha=0.7,
    )
    p.grid.grid_line_color = "lightgrey"
    show(p)
# %%

# %%
# Embed and plot the graph using different layout algorithms for comparison.
# Remove self loops to clean up the graph
G.remove_edges_from(nx.selfloop_edges(G))
# %%
# Filter nodes with degree > 10
G = G.subgraph([n for n, d in G.degree() if d > 100])

# --- 1. Process Layouts ---
print("Computing layouts...")

# PCA Embedding
adj_matrix = nx.to_numpy_array(G)
pca = PCA(n_components=2)
embedding = pca.fit_transform(adj_matrix)
pos_pca = {node: embedding[i] for i, node in enumerate(G.nodes())}

# Laplacian Eigenmaps (Spectral Layout)
# For directed graphs, this uses the Fiedler vector of the symmetrized Laplacian.
pos_spectral = nx.spectral_layout(G)

# Force-Directed Layout
pos_spring = nx.spring_layout(G, k=0.15, iterations=20)
# %%
# --- 2. Plotting ---
print("Plotting layouts...")


def plot_bokeh_graph(G, pos, title):
    """Helper function to plot a graph with Bokeh."""
    plot = figure(
        title=title,
        x_range=(-1.1, 1.1),
        y_range=(-1.1, 1.1),
        tools="pan,wheel_zoom,box_zoom,reset,save",
        width=800,
        height=800,
    )

    # Create a Bokeh graph renderer
    graph_renderer = from_networkx(G, pos, scale=1, center=(0, 0))

    # Set node properties
    node_names = list(G.nodes())
    graph_renderer.node_renderer.data_source.data["name"] = node_names
    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral8[0])
    graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral8[2])

    # Set edge properties
    graph_renderer.edge_renderer.glyph = MultiLine(
        line_color="#CCCCCC", line_alpha=0.8, line_width=1
    )
    graph_renderer.edge_renderer.hover_glyph = MultiLine(
        line_color=Spectral8[2], line_width=2
    )

    # Add node labels
    x, y = zip(*graph_renderer.layout_provider.graph_layout.values())
    source = ColumnDataSource({"x": x, "y": y, "name": node_names})
    labels = LabelSet(
        x="x",
        y="y",
        text="name",
        source=source,
        background_fill_color="white",
        text_font_size="8pt",
        x_offset=5,
        y_offset=5,
        render_mode="canvas",
    )
    plot.add_layout(labels)

    # Add hover tool
    tooltips = [("Name", "@name")]
    plot.add_tools(
        HoverTool(tooltips=tooltips, renderers=[graph_renderer.node_renderer])
    )

    plot.renderers.append(graph_renderer)
    show(plot)


# Plot PCA Embedding
plot_bokeh_graph(G, pos_pca, f"User Interactions in {month_to_analyze} (PCA Embedding)")

# %%
# Plot Laplacian Eigenmaps
plot_bokeh_graph(
    G, pos_spectral, f"User Interactions in {month_to_analyze} (Laplacian Eigenmaps)"
)
# %%

# Plot Force-Directed Layout
plot_bokeh_graph(
    G, pos_spring, f"User Interactions in {month_to_analyze} (Force-Directed Layout)"
)


# %%
