# %%
from supabase import create_client, Client
from typing import Dict, List, Optional, TypedDict
from datetime import datetime, date, timedelta
import pandas as pd
import os
import logging
import time
import json
import shutil
import argparse

# --- Setup (replace with your actual credentials) ---
SUPABASE_URL = "https://fabxmporizzqflnftavs.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhYnhtcG9yaXp6cWZsbmZ0YXZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjIyNDQ5MTIsImV4cCI6MjAzNzgyMDkxMn0.UIEJiUNkLsW28tBHmG-RQDW-I5JNlJLt62CSk9D_qG8"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# --- End Setup ---

# Set up logging
data_dir = "../data"
logs_dir = "../logs"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

log_file = os.path.join(logs_dir, "tweet_fetch.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def cleanup_logs():
    """Clean up logs directory after successful completion"""
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)
        print(f"Cleaned up logs directory: {logs_dir}")
    else:
        print("No logs directory to clean up")


class EnrichedTweetData(TypedDict):
    tweet_id: str
    account_id: str
    username: str
    account_display_name: str
    created_at: datetime
    full_text: str
    retweet_count: int
    favorite_count: int
    reply_to_tweet_id: Optional[str]
    reply_to_user_id: Optional[str]
    reply_to_username: Optional[str]
    quoted_tweet_id: Optional[str]
    conversation_id: Optional[str]


def get_week_dates(year: int, month: int, week_num: int) -> tuple[date, date]:
    """Get start and end dates for a specific week of a month"""
    first_day = date(year, month, 1)
    # Find the first day of the week (assuming week starts on Monday)
    first_monday = first_day - timedelta(days=first_day.weekday())

    # Calculate the start of the requested week
    week_start = first_monday + timedelta(weeks=week_num - 1)
    week_end = week_start + timedelta(days=6)

    return week_start, week_end


def get_tweets_from_week(
    supabase: Client, year: int, month: int, week_num: int, limit: int = 1000
) -> List[EnrichedTweetData]:
    """
    Get enriched tweets from a specific week of a month

    Args:
        supabase: Supabase client
        year: Year (e.g., 2023)
        month: Month (e.g., 11 for November)
        week_num: Week number in month (1-4)
        limit: Maximum number of tweets to return
    """
    week_start, week_end = get_week_dates(year, month, week_num)

    # Convert to datetime strings for the query
    start_datetime = week_start.strftime("%Y-%m-%d 00:00:00")
    end_datetime = week_end.strftime("%Y-%m-%d 23:59:59")

    response = (
        supabase.table("enriched_tweets")
        .select(
            """
            tweet_id,
            account_id,
            username,
            account_display_name,
            created_at,
            full_text,
            retweet_count,
            favorite_count,
            reply_to_tweet_id,
            reply_to_user_id,
            reply_to_username,
            quoted_tweet_id,
            conversation_id
        """
        )
        .gte("created_at", start_datetime)
        .lte("created_at", end_datetime)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

    return response.data


def save_checkpoint(tweets: List[EnrichedTweetData], filename: str) -> None:
    """Save checkpoint of tweets collected so far"""
    os.makedirs(data_dir, exist_ok=True)

    checkpoint_file = os.path.join(data_dir, f"{filename}_checkpoint.parquet")
    df = pd.DataFrame(tweets)
    df.to_parquet(checkpoint_file, index=False)

    # Also save metadata
    metadata = {
        "total_tweets": len(tweets),
        "last_updated": datetime.now().isoformat(),
        "filename": filename,
    }
    metadata_file = os.path.join(data_dir, f"{filename}_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Checkpoint saved: {len(tweets)} tweets to {checkpoint_file}")


# --- Generic date-range variant ---
def get_all_tweets_from_range_paginated(
    supabase: Client,
    start_datetime: datetime,
    end_datetime: datetime,
    batch_size: int = 1000,
    checkpoint_interval: int = 10,
) -> List[EnrichedTweetData]:
    """Fetch *all* enriched tweets between *start_datetime* and *end_datetime* (inclusive) using pagination."""

    start_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

    filename_stub = f"tweets_{start_datetime.strftime('%Y_%m_%d')}_to_{end_datetime.strftime('%Y_%m_%d')}"

    logger.info(f"Starting tweet fetch for {start_str} → {end_str}")

    all_tweets: List[EnrichedTweetData] = []
    offset = 0
    batch_count = 0
    start_time = time.time()

    while True:
        try:
            response = (
                supabase.table("enriched_tweets")
                .select(
                    """
                    tweet_id,
                    account_id,
                    username,
                    account_display_name,
                    created_at,
                    full_text,
                    retweet_count,
                    favorite_count,
                    reply_to_tweet_id,
                    reply_to_user_id,
                    reply_to_username,
                    quoted_tweet_id,
                    conversation_id
                """
                )
                .gte("created_at", start_str)
                .lte("created_at", end_str)
                .order("created_at", desc=True)
                .range(offset, offset + batch_size - 1)
                .execute()
            )
        except Exception as e:
            logger.error(f"Error fetching batch at offset {offset}: {e}")
            break

        batch_tweets = response.data
        if not batch_tweets:
            logger.info("No more tweets to fetch")
            break

        all_tweets.extend(batch_tweets)
        batch_count += 1

        logger.info(
            f"Batch {batch_count}: fetched {len(batch_tweets)} tweets (total: {len(all_tweets)})"
        )

        if batch_count % checkpoint_interval == 0:
            save_checkpoint(all_tweets, filename_stub)

        if len(batch_tweets) < batch_size:
            logger.info("Reached end of data (partial batch)")
            break

        offset += batch_size

    total_time = time.time() - start_time
    logger.info(
        f"Completed: {len(all_tweets)} tweets in {total_time:.2f}s "
        f"({len(all_tweets)/total_time:.2f} tweets/sec)"
    )

    return all_tweets


def get_tweets_by_ids(
    supabase: Client, tweet_ids: List[str]
) -> Dict[str, EnrichedTweetData]:
    """
    Get enriched tweets that exist for the given tweet IDs.
    Returns dict of tweet_id -> tweet data
    """
    if not tweet_ids:
        return {}

    response = (
        supabase.table("enriched_tweets")
        .select(
            """
            tweet_id,
            account_id,
            username,
            account_display_name,
            created_at,
            full_text,
            retweet_count,
            favorite_count,
            reply_to_tweet_id,
            reply_to_user_id,
            reply_to_username,
            quoted_tweet_id,
            conversation_id
        """
        )
        .in_("tweet_id", tweet_ids)
        .execute()
    )

    return {row["tweet_id"]: row for row in response.data if isinstance(row, dict)}


def save_tweets_to_parquet(tweets: List[EnrichedTweetData], filename: str) -> None:
    """Save tweets to parquet file"""
    if not tweets:
        logger.warning("No tweets to save")
        return

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame(tweets)

    # Save to parquet
    filepath = os.path.join(data_dir, f"{filename}.parquet")
    df.to_parquet(filepath, index=False)

    logger.info(f"Saved {len(tweets)} tweets to {filepath}")
    logger.info(f"DataFrame shape: {df.shape}")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Fetch tweets from Supabase within a date range"
    )
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--checkpoint-interval", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d") + timedelta(
        hours=23, minutes=59, seconds=59
    )

    try:
        tweets = get_all_tweets_from_range_paginated(
            supabase,
            start_dt,
            end_dt,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval,
        )

        if tweets:
            filename = f"tweets_{start_dt.strftime('%Y_%m_%d')}_to_{end_dt.strftime('%Y_%m_%d')}"
            save_tweets_to_parquet(tweets, filename)
            save_checkpoint(tweets, filename)

            logger.info("Sample tweets:")
            for tweet in tweets[:3]:
                logger.info(f"Tweet ID: {tweet['tweet_id']}")
                logger.info(
                    f"Author: @{tweet['username']} ({tweet['account_display_name']})"
                )
                logger.info(f"Created: {tweet['created_at']}")
                logger.info(f"Text: {tweet['full_text'][:100]}...")
                logger.info(f"Quoted Tweet ID: {tweet.get('quoted_tweet_id', 'None')}")
                logger.info(f"Conversation ID: {tweet.get('conversation_id', 'None')}")
                logger.info("-" * 50)

            logger.info("Script completed successfully, cleaning up logs...")
            cleanup_logs()
        else:
            logger.warning("No tweets found for the given date range.")

    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        logger.info("Logs preserved for debugging")
        raise

# %%
