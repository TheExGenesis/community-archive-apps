# %%
import pandas as pd

df = pd.read_csv("filtered_tweets.csv")


# %%
# acccount_id to username dict
account_id_to_username = dict(
    zip(df[df.username.notna()].account_id, df[df.username.notna()].username)
)
# %%
df["username"] = df["account_id"].map(account_id_to_username)
# %%
df


# %%
# sample 100 tweets and put their full_text into a string of 100 examples
def get_example_str(n_examples: int = 100):
    examples = df.sample(n_examples)["full_text"].tolist()
    examples_str = "\n---\n".join(examples)
    return examples_str


examples_str = get_example_str()
# %%
print(examples_str)
# %%
import os
from openai import OpenAI

import asyncio
from typing import List, Optional


def get_openrouter_client():
    """Get OpenRouter client with proper configuration"""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://community-archive.org",
            "X-Title": "community-archive",
        },
    )


def query_llm(
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
    client = get_openrouter_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


PROMPT = """
<EXAMPLES>
{examples}
</EXAMPLES>

 
I've given you some examples of some good TPOT tweets. I'm going to ask you to write a tweet about a topic and I want you to imitate their POV and style. No talking just tweet. No hashtags!

Topic: {topic}.
"""
# %%
from pprint import pprint

print(
    query_llm(
        PROMPT.format(examples=get_example_str(), topic="advanced rationalist dharma"),
        model="openai/gpt-4o-2024-11-20",
    )
)

# query_llm(PROMPT.format(examples=examples_str, topic="palantir"))

# %%
tpot_topics = [
    "polyamory",
    "rationalist dating",
    "jhana",
    "qualia",
    "memetics",
    "network states",
    "pop up villages",
    "digital nomadism",
    "Xerox PARC",
    "Georgism",
    "AI risk",
    "AI optimism",
    "community building",
    "metamodernism",
    "Alexander technique",
    "Vajrayana Buddhism",
    "Relationships" "Parenting",
    "Daoism",
    "crypto",
    "collective sensemaking",
    "pronatalism",
]

tweets = [
    query_llm(
        PROMPT.format(examples=get_example_str(100), topic=topic),
        model="openai/gpt-4o-mini-2024-07-18	",
    )
    for topic in tpot_topics
]
# %%
for tweet in tweets:
    pprint(tweet)
    print("\n---\n")
# %%
# save to file
with open("many_shot_tpot_tweets-4o-mini.txt", "w") as f:
    f.write("\n---\n".join(tweets))
# %%
