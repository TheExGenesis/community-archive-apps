# %%
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import re
import os
from supabase import create_client
import dotenv

dotenv.load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase_client = create_client(supabase_url, supabase_key)


def get_usernames():
    # Query both accounts and mentioned_users tables
    accounts = supabase_client.table("account").select("*").execute()
    mentioned = supabase_client.table("mentioned_users").select("*").execute()

    # Build mapping from account_id to username
    id_to_username = {}

    # Add accounts
    for account in accounts.data:
        id_to_username[account["account_id"]] = account["username"]

    # Add mentioned users (user_id is the account_id, screen_name is the username)
    for user in mentioned.data:
        id_to_username[user["user_id"]] = user["screen_name"]

    return id_to_username


def clean_tweet_text(text):
    # Remove "This Post is from a suspended account. {learnmore}"
    text = re.sub(r"This Post is from a suspended account.*", "", text)
    # Remove links of the form "https://t.co/{id}"
    text = re.sub(r"https://t\.co/\w+", "", text)

    # Remove retweet prefix "RT @username:"
    text = re.sub(r"^RT @[A-Za-z0-9_]+: ", "", text)

    # Remove "@" mentions and extra whitespace at the beginning
    text = re.sub(r"^(\s*@\w+\s*)+", "", text)

    return text.strip()  # Remove leading/trailing whitespace


def chunk_document(doc, max_words=32):
    # naive approach:
    # 1) split by sentences
    # 2) split further by newlines
    # 3) split big stuff into 32-word lumps
    # feel free to replace w/ something more robust
    chunks = []
    sents = doc.split(".")  # obviously leaves the '.' behind
    for sent in sents:
        subchunks = sent.split("\n")
        for sc in subchunks:
            words = sc.strip().split()
            i = 0
            while i < len(words):
                chunk = words[i : i + max_words]
                chunks.append(" ".join(chunk))
                i += max_words
    return [c.strip() for c in chunks if c.strip()]


def build_sparse_embeddings(chunks: list[str]) -> tuple[sp.csr_matrix, TfidfVectorizer]:
    """Build sparse TF-IDF embeddings for text chunks.

    Returns:
        tuple: (sparse embeddings matrix, fitted vectorizer)
    """
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(chunks)
    # Convert to float16 while keeping sparsity
    embeddings.data = embeddings.data.astype(np.float16)
    return sp.csr_matrix(embeddings, dtype=np.float16), vectorizer


from tqdm import tqdm


def prune_edges(sim: sp.csr_matrix, threshold: float, max_edges: int) -> sp.csr_matrix:
    # Convert to COO for easier filtering
    sim_coo = sim.tocoo()

    # if we have too many edges, find threshold that gives us max edges
    if len(sim_coo.data) > max_edges:
        threshold = np.partition(sim_coo.data, -max_edges)[-max_edges]
        print(f"Pruning edges below threshold: {threshold}")

    # prune low values
    mask = sim_coo.data >= threshold
    sim = sp.coo_matrix(
        (sim_coo.data[mask], (sim_coo.row[mask], sim_coo.col[mask])),
        shape=sim_coo.shape,
    ).tocsr()

    return sim


def build_adj_matrix(
    embeddings: sp.csr_matrix, batch_size=1000, threshold=0.1, max_edges=50_000_000
) -> sp.csr_matrix:
    n = embeddings.shape[0]
    # Initialize empty CSR matrix
    sim = sp.csr_matrix((n, n), dtype=np.float32)  # Start with float32

    for i in tqdm(range(0, n, batch_size)):
        batch_end = min(i + batch_size, n)
        # Compute similarities for this batch
        batch_sim = embeddings[i:batch_end].dot(embeddings.T)
        # Filter low values immediately
        batch_sim.data[batch_sim.data < threshold] = 0
        batch_sim.eliminate_zeros()

        # Assign without dtype conversion
        sim[i:batch_end] = batch_sim

    # to normalize columns in csr, easiest is transpose =>
    # row-normalize => transpose

    # Normalize as before
    sim_t = sim.transpose().tocsr()
    row_sums = np.array(sim_t.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1e-9
    for i in range(sim_t.shape[0]):
        start, end = sim_t.indptr[i], sim_t.indptr[i + 1]
        sim_t.data[start:end] /= row_sums[i]

    return sim_t.transpose().tocsr()


def personalized_pagerank(
    a: sp.csr_matrix, p: sp.csr_matrix, alpha, max_iter=18, tol=1e-6
):
    n = a.shape[0]
    pr = np.ones((n, 1)) / n  # Make pr a column vector
    for _ in range(max_iter):

        old = pr
        # spmv op
        pr = (1 - alpha) * a.dot(pr) + alpha * p
        # early stopping
        if np.linalg.norm(pr - old, 1) < tol:
            break
    return pr.ravel()  # Convert back to 1D array for return


class SparseRAG:
    def __init__(self, chunks, path_to_adj_matrix=None):
        self.chunks = chunks
        t0 = time.time()
        self.embeddings, self.vectorizer = build_sparse_embeddings(chunks)
        print(f"Building sparse embeddings: {time.time() - t0:.3f}s")

        if path_to_adj_matrix is None:
            t0 = time.time()
            self.adj_matrix = build_adj_matrix(self.embeddings)
            print(f"Building adjacency matrix: {time.time() - t0:.3f}s")
        else:
            with open(path_to_adj_matrix, "rb") as f:
                self.adj_matrix = pickle.load(f)

    def get_query_sims(self, query_text):
        # Transform query using cached vectorizer into sparse matrix
        query_vec = self.vectorizer.transform([query_text])
        # Compute similarities between query and chunks, keeping sparse
        query_sims = sp.csr_matrix.dot(self.embeddings, query_vec.T)
        # Normalize while keeping sparse structure
        query_sum = query_sims.sum()
        if query_sum == 0:
            query_sum = 1e-9
        query_sims.data /= query_sum
        return sp.csr_matrix(query_sims)

    def query(self, query_text, alpha_local=0.6, k=100):
        # Get query similarities
        t0 = time.time()
        query_sims = self.get_query_sims(query_text)

        # Run modified PageRank
        t0 = time.time()
        pr_scores = personalized_pagerank(self.adj_matrix, query_sims, alpha_local).A1
        print(f"PageRank: {time.time() - t0:.3f}s")

        # Get top results
        t0 = time.time()
        topk = np.argsort(-pr_scores)[:k]
        top_chunks = [self.chunks[i] for i in topk]
        # top_chunks = sorted(top_chunks, key=lambda x: self.chunks.index(x))

        return top_chunks, pr_scores, topk


# %%
# load this json
import json

filepath = "/Users/frsc/Documents/Projects/open-birdsite-db/data/downloads/archives/exgenesis/2024-08-14T08:14:14.000Z.json"
with open(filepath, "r") as f:
    data = json.load(f)

# data['tweets'][0]['tweet'].keys()
# dict_keys(['edit_info', 'retweeted', 'source', 'entities', 'display_text_range', 'favorite_count', 'in_reply_to_status_id_str', 'id_str', 'in_reply_to_user_id', 'truncated', 'retweet_count', 'id', 'in_reply_to_status_id', 'created_at', 'favorited', 'full_text', 'lang', 'in_reply_to_screen_name', 'in_reply_to_user_id_str'])
# %%
import toolz.curried as tz

tweet_texts = [
    clean_tweet_text(tweet["tweet"]["full_text"])
    for tweet in data["tweets"]
    if (
        int(tweet["tweet"]["favorite_count"]) >= 1
        and len(clean_tweet_text(tweet["tweet"]["full_text"])) >= 20
    )
]
# %%
tweet_texts = [t for t in tweet_texts if len(t.strip()) > 0]
chunked_tweet_texts = tz.pipe(
    tweet_texts,
    # tz.map(
    #     lambda x: chunk_document(x) if len(x) > 280 else [x]
    # ),
    # tz.concat,
    list,
)


# %%
# Initialize once
rag = SparseRAG(chunked_tweet_texts)
# %%
# Query multiple times efficiently
query = "active infrence"
out, pr_scores, topk = rag.query(
    query,
    alpha_local=0.6,
    k=50,
)
print(f"Query: {query}")
for i, chunk in enumerate(out):
    print(f"{i+1}. {chunk}")
# %%
# Example of another query reusing the same matrices
out2, pr_scores2, topk = rag.query(
    "what are the main themes?",
    alpha_local=0,
    k=100,
)
print("\nSecond query:")
for i, chunk in enumerate(out2):
    print(f"{i+1}. {chunk}")
# %%
import pandas as pd

pd.set_option("display.max_colwidth", None)

df = pd.read_parquet(
    "/Users/frsc/Documents/Projects/tpot-llm/data/borg-ca-tpot.parquet"
)
# %%
# dict_keys(['account_id', 'created_via', 'username', 'created_at', 'account_display_name', 'num_tweets', 'num_following', 'num_followers', 'num_likes'])
id_name_dict = get_usernames()
# %%
# df.columns
# Index(['account_id', 'full_text', 'favorite_count', 'source', 'processed_text',
#        'processed_token_length'],
#       dtype='object')

rag = SparseRAG(df["processed_text"].tolist(), path_to_adj_matrix="data/adj_matrix.pkl")
# %%
# Query multiple times efficiently
query = "hyperobject hyperobject hyperobject at the end of time"

out, pr_scores, topk = rag.query(
    query,
    alpha_local=0.6,
    k=50,
)
print(f"Query: {query}")
for i, (chunk, idx) in enumerate(zip(out, topk)):
    print(f"{i+1}. {chunk}")
    print(
        f"   Account: {id_name_dict[str(df.iloc[idx]['account_id'])] if str(df.iloc[idx]['account_id']) in id_name_dict else 'Unknown'}, Likes: {df.iloc[idx]['favorite_count']}\n"
    )
# %%
# Example of another query reusing the same matrices
out2, pr_scores2, topk = rag.query(
    "what are the main themes?",
    alpha_local=0,
    k=100,
)
print("\nSecond query:")
for i, (chunk, idx) in enumerate(zip(out2, topk)):
    print(f"{i+1}. {chunk}")
    print(
        f"   Account: {id_name_dict[str(df.iloc[idx]['account_id'])] if str(df.iloc[idx]['account_id']) in id_name_dict else 'Unknown'}, Likes: {df.iloc[idx]['favorite_count']}\n"
    )
# %%
