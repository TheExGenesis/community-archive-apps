# %%
# get all tweets from August 2024
import pandas as pd
import supabase
import os
import dotenv
import tqdm
from utils import parallel_io_with_retry
import numpy as np
from datetime import datetime, timedelta  # Import here

dotenv.load_dotenv()

supabase = supabase.create_client(
    os.getenv("NEXT_PUBLIC_SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE"),
)


# %%


# %% Refactored function to fetch tweets from the same day across previous years (exclude current year)
def fetch_same_day_all_years(month, day):
    """Fetch tweets from the same day across all previous years until no more tweets are found (excluding current year)."""
    all_tweets = []
    year = datetime.now().year - 1  # exclude current year
    print(f"Fetching tweets for {month:02d}-{day:02d} starting from {year}...")
    while year >= 2006:
        try:
            start_dt = datetime(year, month, day)
            end_dt = start_dt + timedelta(days=1)
            start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
            page_size, idx = 1000, 0
            year_batch = []
            while True:
                resp = (
                    supabase.table("enriched_tweets")
                    .select("*")
                    .gte("created_at", start_str)
                    .lt("created_at", end_str)
                    .range(idx, idx + page_size - 1)
                    .execute()
                )
                batch = resp.data or []
                year_batch.extend(batch)
                if len(batch) < page_size:
                    break
                idx += page_size
            if not year_batch and all_tweets:
                break
            all_tweets.extend(year_batch)
        except ValueError:
            pass
        year -= 1
    print(f"Finished fetching. Total tweets found: {len(all_tweets)}")
    return all_tweets


import os
import pandas as pd

# %% Initial Data Fetching
today = datetime.now()
day, month = today.day, today.month
filename = f"tweets_{month}_{day}_raw.csv"
if os.path.exists(filename):
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)
    print("Data loaded successfully.")
else:
    print(f"{filename} not found. Fetching data...")
    df = pd.DataFrame(fetch_same_day_all_years(month, day))
    print(f"Data fetched. Found {len(df)} tweets.")
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}.")

# %% Filter to accounts in public.all_account view
resp = supabase.table("all_account").select("account_id").execute()
valid_ids = {r["account_id"] for r in (resp.data or [])}
df = df[df["account_id"].isin(valid_ids)]
print(f"Count after filtering to valid accounts: {len(df)}")

# %% Filter out RTs and replies
print(f"Initial tweet count: {len(df)}")
df = df[~df["full_text"].str.startswith("RT @", na=False)]
print(f"Count after removing RTs: {len(df)}")
df = df[df["reply_to_tweet_id"].isnull()]
print(f"Count after removing replies: {len(df)}")

# %%
pd.set_option("display.max_colwidth", None)
df.sort_values(by="favorite_count", ascending=False).head(50)
# %%
# make col width unlimitted

# Ensure 'created_at' is datetime
df["created_at"] = pd.to_datetime(df["created_at"])


# %% Calculate Z-score based on previous 100 tweets for each tweet - EFFICIENTLY (Year-Specific)


def fetch_historical_stats_for_account_year(params: dict):
    """
    Fetches historical stats for a specific account *before* a given date within a specific year.
    Expects params dict with 'account_id', 'year', and 'before_timestamp'.
    Returns dict with 'account_id', 'year', 'hist_mean_favs', 'hist_std_favs'.
    """
    account_id = params["account_id"]
    year = params["year"]  # Keep year for merging back
    before_timestamp = params[
        "before_timestamp"
    ]  # Timestamp is start of day (month, day, year)
    limit = 100

    # Convert datetime to string format Supabase expects (UTC ISO 8601)
    before_timestamp_str = before_timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    if before_timestamp_str.endswith("+00:00"):
        before_timestamp_str = before_timestamp_str[:-6] + "Z"

    try:
        response = (
            supabase.table("enriched_tweets")
            .select("favorite_count")
            .eq("account_id", account_id)
            .lt("created_at", before_timestamp_str)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        favs = [item["favorite_count"] for item in response.data]
    except Exception as e:
        print(
            f"Error fetching historical favs for account {account_id}, year {year}, before {before_timestamp_str}: {e}"
        )
        favs = []

    if len(favs) < 2:
        mean = favs[0] if len(favs) == 1 else 0
        std = 0.0
    else:
        mean = np.mean(favs)
        std = np.std(favs)

    if np.isnan(std):
        std = 0.0

    return {
        "account_id": account_id,
        "year": year,
        "hist_mean_favs": mean,
        "hist_std_favs": std,
    }


# Add year column to DataFrame
# Helper function for timezone handling with datetime objects
# Note: The function add_utc_timezone was removed as it wasn't used here.
# The pandas .dt accessor methods below are the standard way to handle timezones for a Series.

# Ensure 'created_at' column is timezone-aware UTC.
# If the AttributeError persists, ensure the preceding pd.to_datetime call
# uses errors='coerce' to handle potential non-datetime values, e.g.:
# df["created_at"] = pd.to_datetime(df["created_at"], errors='coerce')


# Extract the year component from the UTC datetime.
df["year"] = df["created_at"].dt.year


# Get unique (account_id, year) pairs
unique_account_years = df[["account_id", "year"]].drop_duplicates().to_dict("records")

# Prepare data for parallel fetching
fetch_params = []
for item in unique_account_years:
    acc_id = item["account_id"]
    yr = item["year"]
    # Construct the specific timestamp for the start of the day (month, day) in that year
    try:
        before_timestamp = datetime(yr, month, day).replace(
            tzinfo=None
        )  # Naive initially
        fetch_params.append(
            {"account_id": acc_id, "year": yr, "before_timestamp": before_timestamp}
        )
    except ValueError:
        # Handle potential invalid dates like Feb 29 in non-leap year
        print(
            f"Skipping invalid date for historical stats: {yr}-{month}-{day} for account {acc_id}"
        )

print(
    f"Fetching year-specific historical stats for {len(fetch_params)} unique account-year pairs..."
)

# Run in parallel
results_list = parallel_io_with_retry(
    func=fetch_historical_stats_for_account_year,  # Use the new function
    data=fetch_params,
    max_workers=5,  # Reduced workers to alleviate potential resource/rate limits
    max_retries=3,
    delay=1,
)

# Process results into a DataFrame for easy merging
hist_stats_list = [
    res for res in results_list if res
]  # Filter out potential None results
hist_stats_df = pd.DataFrame(hist_stats_list)

print(
    f"Successfully fetched year-specific historical stats for {len(hist_stats_df)} account-year pairs."
)

# Merge historical stats back into the main DataFrame using year and account_id
if not hist_stats_df.empty:
    # Ensure columns exist before merge (in case they were dropped previously)
    if "hist_mean_favs" in df.columns:
        df = df.drop(columns=["hist_mean_favs"])
    if "hist_std_favs" in df.columns:
        df = df.drop(columns=["hist_std_favs"])

    df = pd.merge(df, hist_stats_df, on=["account_id", "year"], how="left")
    # Fill NaNs for pairs where stats couldn't be fetched
    df["hist_mean_favs"] = df["hist_mean_favs"].fillna(0)
    df["hist_std_favs"] = df["hist_std_favs"].fillna(0)
else:
    print(
        "Warning: No year-specific historical stats were fetched. Filling with zeros."
    )
    df["hist_mean_favs"] = 0.0
    df["hist_std_favs"] = 0.0


# %% Calculate z-score using YEAR-SPECIFIC historical stats
epsilon = 1e-6
# Ensure the z-score column is dropped if it exists from a previous run
if "hist_fav_z_score" in df.columns:
    df = df.drop(columns=["hist_fav_z_score"])

df["hist_fav_z_score"] = 0.0  # Initialize column
mask = df["hist_std_favs"] >= epsilon
df.loc[mask, "hist_fav_z_score"] = (
    np.log1p(df.loc[mask, "favorite_count"]) - df.loc[mask, "hist_mean_favs"]
) / df.loc[mask, "hist_std_favs"]


# %% Filter for top 90th percentile of FAVORITE COUNT per account
# Calculate the 90th percentile favorite count threshold within each account group
# Drop previous percentile column if exists
if "account_fav_90p" in df.columns:
    df = df.drop(columns=["account_fav_90p"])
# Drop the old z-score percentile column if it exists
if "account_hist_z_score_90p" in df.columns:
    df = df.drop(columns=["account_hist_z_score_90p"])

fav_90p = df.groupby("account_id")["favorite_count"].quantile(0.9).reset_index()
fav_90p = fav_90p.rename(columns={"favorite_count": "account_fav_90p"})

# Merge the percentile threshold back
df = pd.merge(df, fav_90p, on="account_id", how="left")

# Filter the DataFrame
df_top_90p = df[df["favorite_count"] >= df["account_fav_90p"]].copy()

# Display some results (update columns)
print("Top 90th Percentile Tweets by Favorite Count (per account):")
print(
    df_top_90p[
        [
            "username",
            "created_at",
            "favorite_count",
            # "hist_mean_favs", # Optional: keep if still relevant
            # "hist_std_favs", # Optional: keep if still relevant
            # "hist_fav_z_score", # Optional: keep if still relevant
            "account_fav_90p",  # Show the percentile threshold
        ]
    ]
    .sort_values(by=["username", "favorite_count"], ascending=[True, False])
    .head(10)
)

# %%
# Fetch quote tweet counts for the tweets in the filtered DataFrame
# (Use the filtered df_top_90p)
tweet_ids = df_top_90p["tweet_id"].tolist()


# Function to fetch quotes for a single batch of tweet IDs
# Input: batch_tweet_ids: List[str]
# Output: List[Dict[str, str]] (list of {'quoted_tweet_id': id})
def fetch_quotes_for_batch(batch_tweet_ids):
    batch_quote_entries = []
    page_size = 1000
    start = 0
    while True:
        # Retry logic is now handled by parallel_io_with_retry
        response = (
            supabase.table("quote_tweets")
            .select("quoted_tweet_id")
            .in_("quoted_tweet_id", batch_tweet_ids)
            .range(start, start + page_size - 1)
            .execute()
        )
        batch_data = response.data
        batch_quote_entries.extend(batch_data)
        if len(batch_data) < page_size:
            break
        start += page_size
    return batch_quote_entries


# Prepare batches
batch_size = 100  # Keep the same batch size for input to parallel function
batches = [tweet_ids[i : i + batch_size] for i in range(0, len(tweet_ids), batch_size)]

# Run in parallel
print(f"Fetching quote counts for {len(tweet_ids)} tweets in {len(batches)} batches...")
results_list = parallel_io_with_retry(
    func=fetch_quotes_for_batch,
    data=batches,
    max_workers=5,  # Reduce workers slightly to avoid overwhelming DB
    max_retries=5,
    delay=2,
)

# Combine results from all batches
all_quote_entries = []
for batch_result in tqdm.tqdm(results_list, desc="Combining results"):
    if batch_result:  # Handle potential None results from failed retries
        all_quote_entries.extend(batch_result)


# Convert fetched data to a DataFrame
if all_quote_entries:
    quotes_df = pd.DataFrame(all_quote_entries)
    # Group by quoted_tweet_id and count occurrences
    quote_counts_df = (
        quotes_df.groupby("quoted_tweet_id")
        .size()
        .reset_index(name="quote_tweet_count")
    )
    quote_counts_df = quote_counts_df.rename(columns={"quoted_tweet_id": "tweet_id"})
else:
    # Handle case where no quote tweets were found for any tweet_id
    quote_counts_df = pd.DataFrame(columns=["tweet_id", "quote_tweet_count"])
# %%
# Merge quote counts into the filtered DataFrame
# Use suffixes to avoid duplicate column names if necessary
df_top_90p = pd.merge(df_top_90p, quote_counts_df, on="tweet_id", how="left")

# Fill NaN values in quote_tweet_count with 0 (tweets that were never quoted)
df_top_90p["quote_tweet_count"] = df_top_90p["quote_tweet_count"].fillna(0).astype(int)

# %%
# Display tweets sorted by quote count (from the filtered set)
print("\nTop 90p Historical Z-score tweets, sorted by quote count:")
print(
    df_top_90p[
        [
            "username",
            "created_at",
            "full_text",
            "favorite_count",
            # "hist_fav_z_score", # Removed z-score
            "quote_tweet_count",
        ]
    ]
    .sort_values(by="quote_tweet_count", ascending=False)
    .head(20)
)


# %%
# Save the filtered DataFrame with quote counts
df_top_90p.to_csv(
    f"tweets_{month}_{day}_top90p_favcount_with_quotes.csv",
    index=False,  # Updated filename
)

# %%
