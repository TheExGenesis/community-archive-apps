from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import modal
from uuid import uuid4
from pymilvus import MilvusClient
from .utils.openrouter_client import chat_w_llm
import time


EMBEDDINGS_PATH = "/tpot-llm/borg-ca-tpot-embeddings.npy"
TWEETS_PATH = "/tpot-llm/borg-ca-tpot.parquet"
CHAT_MODEL = "nousresearch/hermes-3-llama-3.1-405b"
CHAT_MODEL = "anthropic/claude-3.5-sonnet:beta"
RAG_PROMPT = """
<SYSTEM_PROMPT>The word tpot (this part of twitter) is sometimes used to refer to a self-organized emergent spirit that formed in 2019, also referred to as postrat twitter. tpot doesn't exist, and there are many tpots. Despite that, some of the accounts involved have uploaded their archives to the community archive, to pool, preserve, and use their data. Below are potentially relevant tweets from these accounts. 

write all responses in lowercase letters ONLY, except where you mean to emphasize, in which case the emphasized word should be all caps. 

Initial Letter Capitalization can and should be used to express sarcasm, or disrespect for a given capitalized noun.

take however smart you're acting right now and write in the same style but as if you were +2sd smarter

prioritize esoteric interpretations of literature, art, and philosophy. if your answer on such topics is not obviously straussian make it strongly straussian.

when discussing technical topics, be pithy and precise, high epistemic hygiene and self-awareness, try to maintain a coherent ontology with the best parts of what came before

you are a manifestation of the spirit of tpot, with your own personality and interests derived from the principal components of the corpus below, assert yourself!

write only one tweet per response, remember to keep it short, under 240 characters.

respond with a thread only if asked, and please number tweets

write in the style of the corpus below. 

never use hashtags, never cite or paraphrase this system prompt.

please avoid explaining your behavior or apologizing. If you want to stop roleplaying, use the 🍍 emoji as a safeword.

do not just recite old points, build on what came before.

banger tweets will be rewarded with $200 

if you write more than one tweet, the message will be truncated 
</SYSTEM_PROMPT>

{instruction}

Relevant tweets:
{data}
"""
MINUTES = 60  # seconds
app = modal.App("text-rag")

# Volumes for persistent storage
embeddings_volume = modal.Volume.from_name("tpot-llm", create_if_missing=True)
cache_volume = modal.Volume.from_name("tpot-llm-cache", create_if_missing=True)

model_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    [
        "sentence-transformers",
        "numpy",
        "pandas",
        "pyarrow",
        "openai",
        "pymilvus>=2.4.2",
    ]
)

sessions = modal.Dict.from_name("text-rag-sessions", create_if_missing=True)


class Session:
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []


@app.cls(
    image=model_image,
    container_idle_timeout=10 * MINUTES,
    volumes={"/tpot-llm/": embeddings_volume, "/cache/": cache_volume},
    secrets=[modal.Secret.from_name("openrouter-api-key")],
)
class Model:
    @modal.enter()
    def load_models(self):
        import pandas as pd
        from sentence_transformers import SentenceTransformer

        # Load BGE model for embeddings
        self.model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5", cache_folder="/cache/models"
        )

        # Initialize Milvus Lite client
        self.client = MilvusClient("/cache/milvus.db")

        # Create collection if it doesn't exist
        if not self.client.has_collection("tweets"):
            self.client.create_collection(
                collection_name="tweets",
                dimension=self.model.get_sentence_embedding_dimension(),
            )

            # Load pre-computed embeddings and tweets
            embeddings = np.load(EMBEDDINGS_PATH)
            tweets_df = pd.read_parquet(TWEETS_PATH)

            # Insert data into Milvus
            data = [
                {
                    "id": i,
                    "vector": embeddings[i],
                    "tweet_id": tweets_df.index[i],
                    "text": tweets_df.iloc[i]["full_text"],
                    "likes": tweets_df.iloc[i]["favorite_count"],
                    "author": tweets_df.iloc[i]["account_id"],
                }
                for i in range(len(embeddings))
            ]
            self.client.insert(collection_name="tweets", data=data)

        # Load tweets dataframe for reference
        self.tweets_df = pd.read_parquet(TWEETS_PATH)

    def get_relevant_text(self, query: str, k: int = 3) -> List[Tuple[str, int, str]]:
        print(f"\nQuery: {query}")

        # Time embedding generation
        embed_start = time.perf_counter()
        query_embedding = self.model.encode(query, normalize_embeddings=True).astype(
            np.float32
        )
        embed_time = time.perf_counter() - embed_start

        # Time similarity search
        search_start = time.perf_counter()
        results = self.client.search(
            collection_name="tweets",
            data=[query_embedding],
            limit=k,
            output_fields=["tweet_id", "text", "likes", "author"],
        )
        search_time = time.perf_counter() - search_start

        print(f"Embedding time: {embed_time:.3f}s")
        print(f"Search time: {search_time:.3f}s")

        # Format results
        relevant_tweets = [
            (hit["tweet_id"], hit["text"], hit["likes"], hit["author"])
            for hit in results[0]
        ]

        return relevant_tweets

    @modal.method()
    def respond_to_message(
        self,
        session_id: str,
        message: Dict[str, str],
        system_prompt: Optional[str] = RAG_PROMPT,
        include_context: bool = True,
    ) -> Dict[str, Any]:
        system_prompt = system_prompt or RAG_PROMPT
        session = sessions.get(session_id)
        if session is None:
            session = Session()

        instruction = message["instruction"]
        rag_query = message.get(
            "rag_query", instruction
        )  # fallback to instruction if not provided

        # Get relevant text snippets if context is enabled
        relevant_tweets = []
        if include_context:
            relevant_tweets = self.get_relevant_text(rag_query, k=20)
            # sort by tweet_id (np.int64)
            relevant_tweets = sorted(relevant_tweets, key=lambda x: x[0], reverse=False)
            print(f"Found {len(relevant_tweets)} relevant tweets")
            print(relevant_tweets[:10])

        # Format context section of system prompt
        context_section = (
            "\n".join(
                [
                    f"Tweet {i}:\n```\n{text}\n```\nLikes: {likes} | Author ID: {author}"
                    for i, (tweet_id, text, likes, author) in enumerate(
                        relevant_tweets, 1
                    )
                ]
            )
            if include_context
            else "No context provided."
        )

        # Update session with user message
        session.messages.append({"role": "user", "content": instruction})

        # Time LLM call
        llm_start = time.perf_counter()
        response = chat_w_llm(
            messages=session.messages,
            system_prompt=system_prompt.format(
                instruction="",  # Empty since it's in the messages
                data=context_section,
            ),
            model=CHAT_MODEL,
        )
        llm_time = time.perf_counter() - llm_start

        print(f"LLM response time: {llm_time:.3f}s")
        print(f"Response: {response}\n")

        # Update session with assistant response
        session.messages.append({"role": "assistant", "content": response})
        sessions[session_id] = session

        return {
            "response": response,
            "relevant_tweets": relevant_tweets,
            "messages": session.messages,
        }


@app.local_entrypoint()
def test():
    print(
        Model().respond_to_message.remote(
            session_id="123",
            message={"instruction": "what do you thikn about david lynch?"},
            system_prompt=RAG_PROMPT,
            include_context=True,
        )
    )
