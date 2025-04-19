# %%
{
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

# %%
import os
from supabase import create_client
import dotenv
from toolz import pipe, curry
from typing import List, Dict, Any
import seaborn as sns

dotenv.load_dotenv()


# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase_client = create_client(supabase_url, supabase_key)


# Query top 1000 tweets by favorite_count
def get_top_tweets(limit: int = 1000):
    all_tweets = []
    batch_size = 100

    for offset in range(0, limit, batch_size):
        batch_limit = min(batch_size, limit - offset)
        response = (
            supabase_client.table("tweets_enriched")
            .select("*")
            .order("favorite_count", desc=True)
            .limit(batch_limit)
            .offset(offset)
            .execute()
        )
        all_tweets.extend(response.data)

    return all_tweets


def get_accounts_for_tweets(tweets):
    # Get unique account IDs from tweets
    account_ids = list(set(tweet["account_id"] for tweet in tweets))

    # Query accounts table for these IDs
    response = (
        supabase_client.table("account")
        .select("*")
        .in_("account_id", account_ids)
        .execute()
    )
    return response.data


# Fetch the tweets and corresponding accounts
top_tweets = get_top_tweets()
accounts = get_accounts_for_tweets(top_tweets)
# %%
# Print first few tweets to verify
for tweet in top_tweets[:50]:
    print(
        f"Favorites: {tweet['favorite_count']}, User: {tweet['username']}, Created At: {tweet['created_at']}, Text: {tweet['full_text']}"
    )

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Make a high resolution pie chart of tweets by username
tweets_df = pd.DataFrame(top_tweets)
# Convert accounts to DataFrame and calculate average likes per tweet
accounts_df = pd.DataFrame(accounts)
# %%
# Plot average likes per tweet
accounts_df["avg_likes"] = accounts_df["num_likes"] / accounts_df["num_tweets"]
top_20_avg_likes = accounts_df["avg_likes"].nlargest(20)

plt.figure(figsize=(15, 8))
plt.bar(range(len(top_20_avg_likes)), top_20_avg_likes)
plt.xticks(
    range(len(top_20_avg_likes)),
    accounts_df.loc[top_20_avg_likes.index, "username"],
    rotation=45,
    ha="right",
)
plt.title("Top 20 Accounts by Average Likes per Tweet")
plt.ylabel("Average Likes per Tweet")
plt.tight_layout()
# %%
# Plot total likes
plt.figure(figsize=(15, 8))
top_20_total_likes = accounts_df["num_likes"].nlargest(20)
plt.bar(range(len(top_20_total_likes)), top_20_total_likes)
plt.xticks(
    range(len(top_20_total_likes)),
    accounts_df.loc[top_20_total_likes.index, "username"],
    rotation=45,
    ha="right",
)
plt.title("Top 20 Accounts by Total Likes")
plt.ylabel("Total Likes")
plt.tight_layout()
# %%

# %%
# Get value counts and calculate percentages
value_counts = tweets_df.iloc[:100]["username"].value_counts()
percentages = value_counts / len(tweets_df.iloc[:100]) * 100

# Get top 10 users and combine rest into 'misc'
top_10_users = percentages.nlargest(10)
misc_pct = percentages[~percentages.index.isin(top_10_users.index)].sum()
plot_data = pd.concat([top_10_users, pd.Series({"misc": misc_pct})])

# Use a sophisticated color palette with rich jewel tones and pastels
colors = [
    "#2E1F27",
    "#854D27",
    "#DD7230",
    "#F4C95D",
    "#E7D7C1",
    "#8B9474",
    "#735D78",
    "#BD4F6C",
    "#4C956C",
    "#31708E",
    "#687864",
]

fig = go.Figure(
    data=[
        go.Pie(
            labels=plot_data.index,
            values=plot_data.values,
            textinfo="percent+label",
            textfont_size=20,
            marker=dict(colors=colors),
            hovertemplate="%{label}<br>%{percent}<extra></extra>",
        )
    ]
)

fig.update_layout(
    title=dict(text="Authors of Top 100 Bangers", font=dict(size=32), y=0.95),
    width=1200,
    height=800,
    showlegend=True,
    legend=dict(
        font=dict(size=16),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        x=1.1,
        y=0.5,
    ),
)

fig.show()

# %%
# print usernames
for username in tweets_df["username"].value_counts().index:
    print(username)

# %%
# tweets by rtk254
tweets_df[tweets_df["username"] == "rtk254"].full_text.values
# %%
from toolz import pipe, curry
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go


# Add these new functions:
@curry
def count_bangers_by_account(tweets_df: pd.DataFrame) -> pd.Series:
    """Count number of top tweets per username"""
    return tweets_df["username"].value_counts()


@curry
def merge_with_account_data(
    banger_counts: pd.Series, accounts_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge banger counts with account metadata"""
    return (
        accounts_df.set_index("username")
        .join(banger_counts.rename("num_bangers"))
        .reset_index()
        .fillna(0)
    )


@curry
def calculate_banger_ratio(df: pd.DataFrame) -> pd.Series:
    """Calculate ratio of bangers to total tweets"""
    return (df["num_bangers"] / df["num_tweets"]).mul(100)  # Convert to percentage


@curry
def plot_banger_ratio_hist(df: pd.DataFrame) -> None:
    """Plot bar chart of banger ratios with username labels"""
    # Get top 20 accounts by banger ratio
    top_accounts = df.nlargest(20, "banger_ratio")

    fig = px.bar(
        top_accounts,
        x="username",
        y="banger_ratio",
        title='Top 20 Accounts by "Banger Ratio"',
        labels={
            "banger_ratio": "Percentage of Account's Tweets in Top 1000",
            "username": "Account",
        },
    )

    # Customize layout
    fig.update_layout(
        xaxis_tickangle=45,
        showlegend=False,
        xaxis_title=None,  # Remove x-axis label since it's self-explanatory
    )

    # Add value labels on top of bars
    for i, row in enumerate(top_accounts.itertuples()):
        fig.add_annotation(
            x=i,
            y=row.banger_ratio,
            text=f"{row.banger_ratio:.1f}%",
            showarrow=False,
            yshift=10,
            font=dict(size=10),
        )

    fig.show()


# Create the visualization
pipe(
    tweets_df,
    count_bangers_by_account,
    merge_with_account_data(accounts_df=accounts_df),
    lambda df: df.assign(banger_ratio=df["num_bangers"] / df["num_tweets"] * 100),
    plot_banger_ratio_hist,
)

# Print top 10 accounts by banger ratio
pipe(
    tweets_df,
    count_bangers_by_account,
    merge_with_account_data(accounts_df=accounts_df),
    lambda df: df.assign(banger_ratio=df["num_bangers"] / df["num_tweets"] * 100),
    lambda df: df.nlargest(10, "banger_ratio")[
        ["username", "num_tweets", "num_bangers", "banger_ratio"]
    ],
    lambda df: print(
        "\nTop 10 accounts by banger ratio:\n",
        df.to_string(float_format=lambda x: f"{x:.2f}"),
    ),
)

# %%
# banger mentions
# load banger.csv
# reply_to_username	banger_mentions	example_tweets
# 0	visakanv	82	["@RobertHaisfield @m_ashcroft idk if you're
banger_accs = pd.read_csv("bangers.csv")
# %%
# plot bar chart top 20 with plotly
import plotly.express as px

fig = px.bar(
    banger_accs.nlargest(20, "banger_mentions"),
    x="reply_to_username",
    y="banger_mentions",
    title='Top 20 Accounts by "banger" Mentions',
)

fig.update_layout(
    xaxis_tickangle=45,
    showlegend=False,
    xaxis_title="Username",
    yaxis_title='Number of "banger" Mentions',
)

fig.show()

# %%
# Merge banger mentions with account data
banger_ratio_df = (
    banger_accs.merge(
        accounts_df[["username", "num_tweets"]],
        left_on="reply_to_username",
        right_on="username",
        how="left",
    )
    .assign(
        banger_ratio=lambda df: (
            df["banger_mentions"] / df["num_tweets"] * 1000
        )  # per 1000 tweets
    )
    .fillna(0)
)

# Plot normalized banger mentions
fig = px.bar(
    banger_ratio_df.nlargest(20, "banger_ratio"),
    x="reply_to_username",
    y="banger_ratio",
    title='Top 20 Accounts by "banger" Mentions per 1000 Tweets',
    labels={
        "reply_to_username": "Username",
        "banger_ratio": "Banger Mentions per 1000 Tweets",
    },
)

fig.update_layout(
    xaxis_tickangle=45,
    showlegend=False,
)

# Add value labels on top of bars
for i, row in enumerate(banger_ratio_df.nlargest(20, "banger_ratio").itertuples()):
    fig.add_annotation(
        x=i,
        y=row.banger_ratio,
        text=f"{row.banger_ratio:.1f}",
        showarrow=False,
        yshift=10,
        font=dict(size=10),
    )

fig.show()


# %%
@curry
def get_user_top_tweets(username: str, limit: int = 100) -> List[Dict[Any, Any]]:
    """Get top tweets by favorite count for a specific username

    Args:
        username: str - The username to fetch tweets for
        limit: int - Maximum number of tweets to return

    Returns:
        List[Dict] - List of tweet objects sorted by favorite_count
    """
    batch_size = 100
    all_tweets = []

    for offset in range(0, limit, batch_size):
        batch_limit = min(batch_size, limit - offset)
        response = (
            supabase_client.table("tweets_enriched")
            .select("*")
            .eq("username", username)
            .order("favorite_count", desc=True)
            .limit(batch_limit)
            .offset(offset)
            .execute()
        )
        all_tweets.extend(response.data)

    return all_tweets


# Example usage:
# top_user_tweets = get_user_top_tweets("visakanv")(100)

# %%
