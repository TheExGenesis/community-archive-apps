# %%
# let's get top liked tweets from each account on this day in previous years
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
