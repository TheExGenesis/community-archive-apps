import streamlit as st
from streamlit_tags import st_tags

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import plotly.graph_objects as go
from supabase import create_client, Client
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
import plotly.express as px
import contextvars
import logging
import time
import json
import base64
from pathlib import Path

load_dotenv(".env")
st.set_page_config(layout="wide")

# Add this near the top of the file
logging.basicConfig(level=logging.INFO)

url: str = "https://fabxmporizzqflnftavs.supabase.co"
key: str = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhYnhtcG9yaXp6cWZsbmZ0YXZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjIyNDQ5MTIsImV4cCI6MjAzNzgyMDkxMn0.UIEJiUNkLsW28tBHmG-RQDW-I5JNlJLt62CSk9D_qG8"
)



def format_tweet_count(count):
    digits = len(str(int(count)))
    if digits > 9:
        return f"{count/10**9:.1f}B"
    elif digits > 6:
        return f"{count/10**6:.1f}M"
    elif digits > 3:
        return f"{count/10**3:.1f}K"
    return str(count)


# Add this function to measure execution time
def timeit(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"{func.__name__} took {duration:.2f} seconds")
        return result
    return wrapper


@st.cache_data(ttl=3600)
def fetch_tweets_cached(search_query, start_date, end_date, limit=100):
    logging.info(f"Executing fetch_tweets_cached for query: {search_query}")
    supabase = create_client(url, key)
    result = supabase.rpc(
        "search_tweets",
        {
            "search_query": search_query.replace(" ", "+"),
            "since_date": start_date.isoformat(),
            "until_date": end_date.isoformat(),
            "limit_": limit,
        },
    ).execute()
    df = pd.DataFrame(result.data)
    df['search_word'] = search_query  # Add this line
    df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
    return df

@timeit
async def fetch_tweets(search_words, start_date, end_date, limit=100):
    logging.info(f"Executing fetch_tweets for words: {search_words}")
    results = []
    for word in search_words:
        result = await asyncio.to_thread(fetch_tweets_cached, word, start_date, end_date, limit)
        results.append(result)
    return pd.concat(results, ignore_index=True)



# _streamlit_thread_context = contextvars.ContextVar("streamlit_thread_context")
@st.cache_data(ttl=3600)
def fetch_word_occurrences_cached(word, start_date, end_date, user_ids):
    logging.info(f"Executing fetch_word_occurrences for word: {word}")
    
    supabase = create_client(url, key)
    result = supabase.rpc(
        "word_occurrences",
        {
            "search_word": word,
            "user_ids": user_ids if len(user_ids) > 0 else None,
        },
    ).execute()
    
    filtered_data = [
        item for item in result.data 
        if start_date <= datetime.strptime(item['month'], '%Y-%m').date() <= end_date
    ]
    
    return {word: filtered_data}

@timeit
async def fetch_word_occurrences(search_words, start_date, end_date, user_ids):
    logging.info(f"Executing fetch_word_occurrences for words: {search_words}")
    tasks = [
        asyncio.to_thread(fetch_word_occurrences_cached, word, start_date, end_date, user_ids)
        for word in search_words
    ]
    results = await asyncio.gather(*tasks)
    return {k: v for d in results for k, v in d.items()}


@st.cache_data(ttl=3600)
def fetch_monthly_tweet_counts():
    logging.info("Executing fetch_monthly_tweet_counts")
    supabase = create_client(url, key)
    result = supabase.rpc('get_monthly_tweet_counts').execute()
    df = pd.DataFrame(result.data)
    df['month'] = pd.to_datetime(df['month'], utc=True)
    return df


def plot_word_occurrences(word_occurrences_dict, monthly_tweet_counts, normalize):
    logging.info("Executing plot_word_occurrences")
    df_list = []
    for word, result in word_occurrences_dict.items():
        if result:  # Check if result not empty
            df = pd.DataFrame(result)
            df['month'] = pd.to_datetime(df['month'], utc=True)
            df['word'] = word
            df_list.append(df)
    
    if not df_list:  # If no data, return empty figure
        return go.Figure()
    
    df = pd.concat(df_list)
    df = df.merge(monthly_tweet_counts, on='month', how='left')
    
    if normalize:
        df['normalized_count'] = df['word_count'] / df['tweet_count'] * 1000
        y_col, y_title = 'normalized_count', 'Occurrences per 1000 tweets'
    else:
        y_col, y_title = 'word_count', 'Word Count'

    fig = px.line(df, x='month', y=y_col, color='word', 
                  title=f'Word Occurrences Over Time {"(normalized)" if normalize else ""}')
    fig.update_layout(xaxis_title='Month', yaxis_title=y_title)
    fig.update_traces(mode='lines+markers')  # Add markers for selection
    return fig


@st.cache_data(ttl=3600)
def fetch_users():
    logging.info("Executing fetch_users")
    supabase = create_client(url, key)
    result = supabase.table("account").select("account_id", "username").execute()
    return result.data


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_global_stats():
    supabase = create_client(url, key)
    result = supabase.table("global_activity_summary").select("*").order('last_updated', desc=True).limit(1).execute()
    return result.data[0] if result.data else None

st.title("Trends in the Community Archive")

# Fetch global stats
global_stats = fetch_global_stats()

if global_stats:
    total_tweets = format_tweet_count(global_stats['total_tweets'])
    total_accounts = global_stats['total_accounts']

    # Add explanation and link with dynamic stats
    st.markdown(f"""
        This app analyzes trends in the [Community Archive](https://www.community-archive.org/), an open database 
        and API for tweet histories. With over {total_tweets} tweets from {total_accounts} accounts, it enables developers to build advanced search tools, AI-powered apps, and other sensemaking projects.
    """)
else:
    st.markdown("""
        This app analyzes trends in the [Community Archive](https://www.community-archive.org/), an open database 
        and API for tweet histories. It enables developers to build advanced search tools, AI-powered apps, 
        and other innovative projects using social media data.
    """)

# Add a divider for visual separation
st.divider()

default_words = ["ingroup", "postrat", "tpot"]



async def main():
    logging.info("Executing main function")
    if not st.session_state.get("supabase"):
        st.session_state.supabase = create_client(url, key)
    
    
    selection = None
    col1, col2 = st.columns(2)
    
    with col1:
        search_words = st_tags(
            label="Enter search words",
            text="Press enter after each word",
            value=default_words,
            suggestions=["meditation", "mindfulness", "retreat"],
            maxtags=10,
            key="search_words",
        )

        # Move advanced options to an expander
        with st.expander("Advanced options"):
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input("Start Date", value=date(2020, 1, 1))
            with date_col2:
                end_date = st.date_input("End Date", value=date.today())

            users = fetch_users()
            user_options = {user["username"]: user["account_id"] for user in users}
            selected_users = st.multiselect("Select Users", options=list(user_options.keys()))
            user_ids = [user_options[user] for user in selected_users]
            normalize = st.checkbox('Normalize by monthly tweet count', value=True)

    # Check if query parameters have changed
    query_changed = (
        "prev_search_words" not in st.session_state
        or "prev_start_date" not in st.session_state
        or "prev_end_date" not in st.session_state
        or "prev_user_ids" not in st.session_state
        or search_words != st.session_state.get("prev_search_words")
        or start_date != st.session_state.get("prev_start_date")
        or end_date != st.session_state.get("prev_end_date")
        or user_ids != st.session_state.get("prev_user_ids")
    )

    if query_changed or "tweets_df" not in st.session_state:
        if search_words:
            with st.spinner("Fetching data..."):
                tweets_task = asyncio.create_task(fetch_tweets(search_words, start_date, end_date))
                word_occurrences_task = asyncio.create_task(fetch_word_occurrences(search_words, start_date, end_date, user_ids))
                
                st.session_state.tweets_df = await tweets_task
                st.session_state.word_occurrences_dict = await word_occurrences_task
                st.session_state.monthly_tweet_counts = fetch_monthly_tweet_counts()

                # Update previous query parameters
                st.session_state.update({
                    "prev_search_words": search_words,
                    "prev_start_date": start_date,
                    "prev_end_date": end_date,
                    "prev_user_ids": user_ids
                })
        else:
            st.session_state.tweets_df = pd.DataFrame()
            st.session_state.word_occurrences_dict = {}
            st.session_state.monthly_tweet_counts = fetch_monthly_tweet_counts()

    if "tweets_df" in st.session_state:
        tweets_df = st.session_state.tweets_df
        word_occurrences_dict = st.session_state.word_occurrences_dict
        monthly_tweet_counts = st.session_state.monthly_tweet_counts

        with col1:
            st.subheader("Keyword Trends")
            if word_occurrences_dict:
                fig = plot_word_occurrences(
                    word_occurrences_dict, monthly_tweet_counts, normalize
                )
                st.info("Drag horizontally on the graph to filter tweets in the right column.")
                selection = st.plotly_chart(fig, use_container_width=True, key="word_occurrences", selection_mode='box', on_select="rerun")
            else:
                st.write("No data to display. Please enter search words.")

        with col2:
            st.subheader("Related Tweets")
            tweet_container = st.container()
            tweet_container.markdown(
                """
                <style>
                [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
                    height: 80vh;
                    overflow-y: auto;
                }
                .tweet-container {
                    display: flex;
                    align-items: flex-start;
                    margin-bottom: 20px;
                }
                .tweet-avatar {
                    width: 48px;
                    height: 48px;
                    border-radius: 50%;
                    margin-right: 10px;
                }
                .tweet-content {
                    flex: 1;
                }
                .tweet-content a { color: inherit; text-decoration: none; }
                </style>
                """,
                unsafe_allow_html=True
            )
            with tweet_container:
                if search_words:
                    tabs = st.tabs(search_words)
                    for word, tab in zip(search_words, tabs):
                        with tab:
                            if selection and selection['selection']['points']:
                                selected_dates = [pd.to_datetime(point['x']) for point in selection['selection']['points']]
                                start_date = min(selected_dates).date()
                                end_date = max(selected_dates).date()
                                word_tweets = fetch_tweets_cached(word, start_date, end_date)
                            else:
                                if 'search_word' in tweets_df.columns:
                                    word_tweets = tweets_df[tweets_df['search_word'] == word]
                                else:
                                    st.error("'search_word' column not found in tweets DataFrame. Please check the data fetching process.")
                                    word_tweets = pd.DataFrame()  # Empty DataFrame as fallback

                            st.write(f"Showing tweets for '{word}'")
                            if word_tweets.empty:
                                st.write("No tweets found")
                            else:
                                for _, tweet in word_tweets.iterrows():
                                    tweet_url = f"https://twitter.com/i/web/status/{tweet['tweet_id']}"
                                    highlighted_text = tweet['full_text'].replace(word, f"<b>{word}</b>")
                                    avatar_url = tweet.get('avatar_media_url', '')
                                    st.markdown(
                                        f"""
                                        <div class="tweet-container">
                                            <div class="tweet-avatar">
                                                <img src="{avatar_url}" style="width:100%;height:100%;border-radius:50%;" onerror="handleImageError(this)">
                                            </div>
                                            <div class="tweet-content">
                                                <b>@{tweet['username']}</b> - <a href="{tweet_url}" target="_blank" style="color: inherit; text-decoration: none;">{tweet['created_at']}</a>
                                                <br>
                                                {highlighted_text}
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                    st.markdown("---")
                else:
                    st.write("No search words entered. Please enter words to see related tweets.")

# Add this JavaScript function to the page
st.markdown("""
<script>
function handleImageError(img) {
    img.style.display = 'none';
}
</script>
""", unsafe_allow_html=True)

# Load the placeholder image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

placeholder_img = get_base64_of_bin_file('placeholder.jpg')

# Add this JavaScript function and CSS to the page
st.markdown(f"""
<style>
.tweet-avatar {{
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background-image: url(data:image/jpg;base64,{placeholder_img});
    background-size: cover;
}}
</style>
<script>
function handleImageError(img) {{
    img.style.display = 'none';
}}
</script>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    asyncio.run(main())
