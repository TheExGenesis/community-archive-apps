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
from typing import List, Dict, Any, Tuple
import seaborn as sns
from tqdm import tqdm
import time
import asyncio
from tqdm.asyncio import tqdm_asyncio
import json

dotenv.load_dotenv()


# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE")
supabase_client = create_client(supabase_url, supabase_key)


# Query top 1000 tweets by favorite_count
def get_top_tweets(limit: int = 1000):
    all_tweets = []
    batch_size = 100
    max_retries = 3
    requests_this_minute = 0
    minute_start = time.time()

    for offset in tqdm(range(0, limit, batch_size)):
        batch_limit = min(batch_size, limit - offset)

        # Check if we need to wait for next minute
        if requests_this_minute >= 60:
            time_elapsed = time.time() - minute_start
            if time_elapsed < 60:
                time.sleep(60 - time_elapsed)
            minute_start = time.time()
            requests_this_minute = 0

        for attempt in range(max_retries):
            try:
                response = (
                    supabase_client.table("tweets_enriched")
                    .select(
                        "username, favorite_count, full_text, created_at, tweet_id, account_id, reply_to_tweet_id, reply_to_username"
                    )
                    .order("favorite_count", desc=True)
                    .limit(batch_limit)
                    .offset(offset)
                    .execute()
                )
                requests_this_minute += 1
                all_tweets.extend(response.data)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to fetch tweets after {max_retries} attempts: {e}")
                    raise
                print(f"Attempt {attempt + 1} failed, retrying...")

    return all_tweets


def get_accounts_for_tweets(tweets):
    # Get unique account IDs from tweets
    account_ids = list(set(tweet["account_id"] for tweet in tweets))

    # Query accounts table for these IDs
    response = supabase_client.table("account").select("*").execute()
    return response.data


# Fetch the tweets and corresponding accounts
top_tweets = get_top_tweets(limit=10000)

# %%
# save to json file
with open("top_tweets.json", "w") as f:
    json.dump(top_tweets, f)


# %%
accounts = get_accounts_for_tweets(top_tweets)
# %%
# Print first few tweets to verify
for tweet in top_tweets[-50:]:
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
# %%
from toolz import pipe, curry
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple
from tqdm.asyncio import tqdm_asyncio


# %%
async def get_user_top_tweets(
    account_id: str, limit: int = 100
) -> List[Dict[Any, Any]]:
    """Get top tweets by favorite count for a specific account_id

    Args:
        account_id: str - The account_id to fetch tweets for
        limit: int - Maximum number of tweets to return

    Returns:
        List[Dict] - List of tweet objects sorted by favorite_count
    """
    batch_size = 100
    all_tweets = []
    max_retries = 3
    for offset in range(0, limit, batch_size):
        batch_limit = min(batch_size, limit - offset)

        for attempt in range(max_retries):
            try:
                response = (
                    supabase_client.table("tweets")
                    .select(
                        "favorite_count, full_text, created_at, tweet_id, account_id, reply_to_tweet_id, reply_to_username"
                    )
                    .eq("account_id", account_id)
                    .order("favorite_count", desc=True)
                    .limit(batch_limit)
                    .offset(offset)
                    .execute()
                )
                all_tweets.extend(response.data)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(
                        f"Failed to fetch tweets for account {account_id} after {max_retries} attempts: {e}"
                    )
                    raise
                print(f"Attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(1)
    print(f"Got {len(all_tweets)} tweets for account {account_id}")
    return all_tweets


async def get_top_n_tweets_for_users(
    account_ids: List[str], n: int = 10, batch_size: int = 10
) -> Tuple[List[Dict[Any, Any]], List[str]]:
    """Get top n tweets for each account_id in parallel batches

    Args:
        account_ids: List[str] - List of account_ids to fetch tweets for
        n: int - Number of top tweets to get per user
        batch_size: int - Number of users to process in parallel

    Returns:
        Tuple[List[Dict], List[str]] - Combined list of top tweets and failed account_ids
    """
    failed_accounts = []
    all_tweets = []

    # Process users in batches to avoid overwhelming the DB
    for i in tqdm(range(0, len(account_ids), batch_size)):
        batch = account_ids[i : i + batch_size]
        tasks = []

        for account_id in batch:
            print(f"Getting top {n} tweets for account {account_id}")
            tasks.append(get_user_top_tweets(account_id, n))

        # Wait for batch to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for account_id, result in zip(batch, results):
            if isinstance(result, Exception):
                print(f"Failed to get tweets for {account_id}: {str(result)}")
                failed_accounts.append(account_id)
            else:
                all_tweets.extend(result)

    return all_tweets, failed_accounts


unique_users = list(accounts_df["account_id"].unique())
all_top_tweets, failed_accounts = asyncio.run(
    get_top_n_tweets_for_users(unique_users[25:], 50)
)
with open("user_top_tweets.json", "w") as f:
    json.dump(all_top_tweets, f)
# %%
# %%
all_top_tweets
# %%
# Convert to DataFrame and sort by favorite count
top_tweets_df = pd.DataFrame(all_top_tweets).sort_values(
    "favorite_count", ascending=False
)
# %%
tweets_df = pd.concat([tweets_df, top_tweets_df])
tweets_df
# %%
from typing import List, Dict, Set
from toolz import curry


@curry
def get_tweets_with_media(tweet_ids: List[str]) -> Set[str]:
    """Check which tweets have associated media

    Args:
        tweet_ids: List[str] - List of tweet IDs to check

    Returns:
        Set[str] - Set of tweet IDs that have media
    """
    # Query tweet_media table for the given tweet IDs
    response = (
        supabase_client.table("tweet_media")
        .select("tweet_id")
        .in_("tweet_id", tweet_ids)
        .execute()
    )

    # Return set of tweet IDs that have media
    return set(item["tweet_id"] for item in response.data)


from typing import List, Dict, Set, Iterator
from toolz import curry, partition_all


@curry
def get_tweets_with_media(tweet_ids: List[str], batch_size: int = 500) -> Set[str]:
    """Check which tweets have associated media, processing in batches

    Args:
        tweet_ids: List[str] - List of tweet IDs to check
        batch_size: int - Size of batches to process (default 500)

    Returns:
        Set[str] - Set of tweet IDs that have media
    """
    media_tweets = set()

    # Process tweet IDs in batches
    for batch in partition_all(batch_size, tweet_ids):
        print(f"Processing batch of size {len(batch)}")
        response = (
            supabase_client.table("tweet_media")
            .select("tweet_id")
            .in_("tweet_id", list(batch))
            .execute()
        )
        media_tweets.update(item["tweet_id"] for item in response.data)

    return media_tweets


tweets_df = tweets_df[tweets_df.reply_to_tweet_id.isna()].drop_duplicates(
    subset="tweet_id"
)
tweets_df
# %%
# Example usage:
tweets_with_media = get_tweets_with_media(tweets_df.tweet_id.tolist())
tweets_w_no_media = tweets_df[~tweets_df.tweet_id.isin(tweets_with_media)]
# %%

# also filter out tweets with links like  https://t.co/g2adKjsds2
tweets_w_no_media = tweets_w_no_media[
    ~tweets_w_no_media.full_text.str.contains(r"https://t.co/\w+")
]

# %%


@curry
def get_tweets_with_urls(tweet_ids: List[str], batch_size: int = 500) -> Set[str]:
    """Check which tweets have associated URLs, processing in batches

    Args:
        tweet_ids: List[str] - List of tweet IDs to check
        batch_size: int - Size of batches to process (default 500)

    Returns:
        Set[str] - Set of tweet IDs that have URLs
    """
    url_tweets = set()

    # Process tweet IDs in batches
    for batch in partition_all(batch_size, tweet_ids):
        print(f"Processing batch of size {len(batch)}")
        response = (
            supabase_client.table("tweet_urls")
            .select("tweet_id")
            .in_("tweet_id", list(batch))
            .execute()
        )
        url_tweets.update(item["tweet_id"] for item in response.data)

    return url_tweets


tweets_with_urls = get_tweets_with_urls(tweets_w_no_media.tweet_id.tolist())
tweets_w_no_urls = tweets_w_no_media[~tweets_w_no_media.tweet_id.isin(tweets_with_urls)]

# %%
# make df columns char limit unlimited
pd.set_option("display.max_colwidth", None)
# %%
tweets_w_no_urls[["username", "favorite_count", "full_text"]]
# %%
filtered_tweets = tweets_w_no_urls[tweets_w_no_urls.favorite_count > 100]
filtered_tweets.to_csv("filtered_tweets.csv")
# %%
import os
from openai import AsyncOpenAI

import asyncio
from typing import List, Optional


async def get_openrouter_client():
    """Get OpenRouter client with proper configuration"""
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://community-archive.org",
            "X-Title": "community-archive",
        },
    )


async def query_llm(
    message: str,
    model: str = "qwen/qwen-2.5-coder-32b-instruct",
    max_tokens: int = 8000,
    temperature: float = 0.0,
) -> str:
    """Query LLM through OpenRouter

    Args:
        message: Prompt to send
        model: Model to use
        max_tokens: Max tokens in response
        temperature: Sampling temperature

    Returns:
        Model response text
    """
    client = await get_openrouter_client()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


TOPIC_PROMPT = """
Examples:
<example>
tweet: "goodbye interesting tweet that refreshed away from the timeline. i’ll never see you again"

answer: {{
    "topic": ["Vanished Tweet"],
    "tone": ["Relatable"]
}}
</example>
<example>
tweet: "my ducks? in a row. ordered. disciplined. behaving predictably.\nyour ducks? scattered. in disarray. waddling aimlessly. desperate for a leader to impose structure.\n\npathetic."

answer: {{
    "topic": ["Ducks in a row"],
    "tone": ["Joke"]
}}
</example>
<example>
tweet: "Airbnb Math:\n\n$40/night x 2 nights = $164	"

answer: {{
    "topic": ["AirBNB"],
    "tone": ["Disatisfied"]
}}
</example>
<example>
tweet: "It's pretty crazy the extent to which people's lives is literally bottleneck by their imagination of what their life can be.\nLike there was tweet floating around about "what hobbies are there past 25 besides cooking and exercising" and someone quote-posting, saying how they	"

answer: {{
    "topic": ["Limitting beliefs", "Hobbies"],
    "tone": ["Empowering"]
}}
</example>
<example>
tweet: "what credit card has the highest possible vendor fees? not necessarily good rewards or anything, just high vendor fees. I need something to use at places that don't accept cash.	"

answer: {{
    "topic": ["Credit Card Fees"],
    "tone": ["Spiteful"]
}}
</example>
<example>
tweet: "BDSM references hit different ever since i accidentally went to a sex dungeon and learned that the venn diagram of "people who are into kink" and "people who are into LARPing and board game nights" is a circle"

answer: {{
    "topic": ["BDSM", "Gaming"],
    "tone": ["Joke"]
}}
</example>
<example>
tweet: "You want to get enlightened? Throw yourself at your fear. Again. And again. And again.\nAll else is commentary."

answer: {{
    "topic": ["Enlightenment", "Fear"],
    "tone": ["Dharma Teacher"]
}}
</example>
<example>
tweet: "being a US citizen should come with a .gov.citizen email address. There should be social media that only US citizens can access. There should be apps that let you in only if you have citizenship"

answer: {{
    "topic": ["US Citizenship"],
    "tone": ["Earnest"]
}}
</example>
<example>
tweet: "tweet: "elite colleges reputation launder the rich students by associating them with the very smart students"

answer: {{
    "topic": ["Reputation Launderring", "Elite Colleges"],
    "tone": ["Insightful"]
}}
</example>

If you were given a list of 1-2 word specific topics and a 1-2 word tone description to write this tweet, what would they be? Be as specific as possible while staying concise. Make topics a list even when it's a single topic. Write in json only.


tweet: {tweet}
"""


async def get_topic(tweet: str) -> Optional[str]:
    """Get topic for a single tweet, handling errors with retries"""
    retries = 4
    for attempt in range(retries):
        try:
            return await query_llm(TOPIC_PROMPT.format(tweet=tweet))
        except Exception as e:
            if attempt == retries - 1:  # Last attempt
                print(f"Error getting topic after {retries} attempts: {e}")
                return None
            print(f"Error getting topic (attempt {attempt + 1}/{retries}): {e}")
            await asyncio.sleep(1)  # Brief delay between retries


async def get_topics(tweets: List[str], batch_size: int = 25) -> List[Optional[str]]:
    """Get topics for multiple tweets concurrently in batches"""
    topics = []
    for i in tqdm(range(0, len(tweets), batch_size)):
        batch = tweets[i : i + batch_size]
        batch_topics = await asyncio.gather(*[get_topic(tweet) for tweet in batch])
        topics.extend(batch_topics)
    return topics


# Example usage
tweet = "RT @jason_koh: I'm a big fan of the @OpenAI API. It's a great way to get started with AI. I've been using it for a few months now and it's been a great experience. I've been able to build a lot of cool stuff with it. I'm excited to see what the future holds for the API. #OpenAI #AI #API"


# Run single topic example
topic = asyncio.run(query_llm(TOPIC_PROMPT.format(tweet=tweet)))
print(f"Single topic: {topic}")

# %%

# %%
# Run batch topic example
topics_strs = asyncio.run(get_topics(filtered_tweets.full_text.tolist()))

# %%
import json

# parse json
topics = []


def parse_topic(topic: str) -> Optional[Dict[str, Any]]:
    if topic:  # Handle None values
        try:
            # Handle both raw JSON and code block wrapped JSON
            cleaned = topic.strip()
            if cleaned.startswith("```"):
                # Extract JSON from code block
                json_str = "\n".join(cleaned.split("\n")[1:-1])
                return json.loads(json_str)
            else:
                return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"Error parsing topic JSON: {e}")
            return None
    else:
        return None


topics = [parse_topic(topic) for topic in topics_strs]
# %%
failed_tweet_ids = [
    tweet_id
    for tweet_id, topic in zip(filtered_tweets.tweet_id.tolist(), topics)
    if topic is None
]
print(f"Failed tweet ids: {len(failed_tweet_ids)}")

# %%
tweets_w_topics = filtered_tweets.copy()
tweets_w_topics["topic"] = [
    topic["topic"] if topic is not None else None for topic in topics
]
tweets_w_topics["tone"] = [
    topic["tone"] if topic is not None else None for topic in topics
]
# %%
tweets_w_topics = tweets_w_topics[
    tweets_w_topics.topic.notna() & tweets_w_topics.account_id.notna()
]

# %%
tweets_w_topics[["username", "favorite_count", "full_text", "topic", "tone"]]

# if topic or tone are strings, make them lists
tweets_w_topics["topic"] = tweets_w_topics["topic"].apply(
    lambda x: [x] if isinstance(x, str) else x
)
tweets_w_topics["tone"] = tweets_w_topics["tone"].apply(
    lambda x: [x] if isinstance(x, str) else x
)

tweets_w_topics[["username", "favorite_count", "full_text", "topic", "tone"]]
# %%
# Flatten topics and count occurrences
all_topics = [topic for topics in tweets_w_topics["topic"] for topic in topics]
topic_counts = pd.Series(all_topics).value_counts()

# Flatten tones and count occurrences
all_tones = [tone for tones in tweets_w_topics["tone"] for tone in tones]
tone_counts = pd.Series(all_tones).value_counts()

print("\nTop 20 Topics:")
print(topic_counts.head(20))

print("\nAll Tones:")
print(tone_counts)

# Optional: Create visualizations
import plotly.express as px

# Topics bar chart (top 20)
fig_topics = px.bar(topic_counts.head(20), title="Top 20 Topics")
fig_topics.show()

# Tones pie chart
fig_tones = px.bar(tone_counts.head(20), title="Distribution of Tones")
fig_tones.show()


# %%
def create_training_jsonl(
    tweets_df: pd.DataFrame, output_file: str = "training_data.jsonl"
):
    """Convert tweets dataframe to JSONL training format

    Args:
        tweets_df: DataFrame with columns [full_text, topic, tone]
        output_file: Path to output JSONL file
    """
    training_examples = []
    for _, row in tweets_df.iterrows():
        # Convert topic list to comma-separated string
        topics_str = ", ".join(row["topic"])

        training_example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a tweet composer who writes engaging tweets based on given topics.",
                },
                {
                    "role": "user",
                    "content": f"Write a tweet about the following topics, with the following tone.\ntone: {', '.join(row['tone']).lower()}\ntopic: {topics_str}",
                },
                {"role": "assistant", "content": row["full_text"]},
            ]
        }
        training_examples.append(training_example)
    return training_examples


# shuffle
tweets_w_topics = tweets_w_topics.sample(frac=1).reset_index(drop=True)
training_examples = create_training_jsonl(tweets_w_topics.iloc[:-100])
validation_examples = create_training_jsonl(tweets_w_topics.iloc[-100:])

# Print first few lines of the training file
with open("banger_train.jsonl", "w") as f:
    for example in training_examples:
        f.write(json.dumps(example) + "\n")

with open("banger_val.jsonl", "w") as f:
    for example in validation_examples:
        f.write(json.dumps(example) + "\n")

# %%
# examples
with open("banger_train.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        print(f"\nExample {i+1}:")
        print(json.dumps(json.loads(line), indent=2))
# %%
# %%
# fine tuning
import openai
from openai import OpenAI
import json
from typing import List, Dict


def validate_and_upload_file(
    training_data: List[Dict], file_name: str = "training_data.jsonl"
) -> str:
    """
    Validate and upload the training file to OpenAI
    Returns the file ID if successful
    """
    # Save the JSONL file
    with open(file_name, "w") as f:
        for entry in training_data:
            f.write(json.dumps(entry) + "\n")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        print("Uploading training file...")
        response = client.files.create(file=open(file_name, "rb"), purpose="fine-tune")
        file_id = response.id
        print(f"File uploaded successfully. File ID: {file_id}")
        return file_id
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        return None


def start_fine_tuning_job(file_id: str, validation_file_id: str, model:str="gpt-4o-mini-2024-07-18") -> str:
    """
    Start a fine-tuning job
    Returns the job ID if successful
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        print("Starting fine-tuning job...")
        response = client.fine_tuning.jobs.create(
            training_file=file_id,
            validation_file=validation_file_id,
            model="gpt-4o-mini-2024-07-18",
            hyperparameters={"n_epochs": 3, "batch_size": 4},
        )
        job_id = response.id
        print(f"Fine-tuning job created. Job ID: {job_id}")
        return job_id
    except Exception as e:
        print(f"Error starting fine-tuning job: {str(e)}")
        return None


def monitor_fine_tuning(job_id: str):
    """Monitor the status of the fine-tuning job"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"Status: {job.status}")
        print(f"Trained tokens: {job.trained_tokens}")
        print(f"Model: {job.fine_tuned_model}")
        return job
    except Exception as e:
        print(f"Error monitoring job: {str(e)}")
        return None


# %%
# Start the fine-tuning process
file_id = validate_and_upload_file(training_examples[:len(training_examples)//10])
validation_file_id = validate_and_upload_file(validation_examples[:len(validation_examples)//10])
# %%
if file_id:
    job_id = start_fine_tuning_job(file_id, validation_file_id, model="openai/gpt-4o-2024-11-2	")
    if job_id:
        print("\nInitial job status:")
        job = monitor_fine_tuning(job_id)

        print("\nYou can monitor the job status using this code:")
        print(
            f"""

# To check status later:
client = OpenAI()
job = client.fine_tuning.jobs.retrieve("{job_id}")
print(f"Status: {job.status}")
print(f"Model: {job.fine_tuned_model}")
        """
        )

# %%
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
job = client.fine_tuning.jobs.retrieve(job_id)
print(f"Status: {job.status}")
print(f"Model: {job.fine_tuned_model}")
# %%
