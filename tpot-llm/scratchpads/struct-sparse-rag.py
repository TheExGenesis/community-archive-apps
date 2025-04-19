import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import re


def build_structural_adjacency(
    tweet_index_for_chunk: list[int],
    reply_map: dict[int, int],
    user_map: dict[int, list[int]],
    n_chunks: int,
    w_reply=1.0,
    w_user=0.5,
):
    """
    build a coo adjacency for structural relationships, then convert to csr.

    tweet_index_for_chunk: list giving the tweet-id (or row-index in a df) from which each chunk came
    reply_map: dict[tweet_id] -> tweet_id to which it replies (or -1 if none)
    user_map: dict[user_id] -> list of tweet_ids for that user
    n_chunks: total number of text chunks
    w_reply: weight for reply edges
    w_user: weight for same-user edges
    """
    rows, cols, data = [], [], []

    # build a mapping from tweet id to list of chunk indices
    from collections import defaultdict

    tweet_to_chunks = defaultdict(list)
    for chunk_idx, tw_id in enumerate(tweet_index_for_chunk):
        tweet_to_chunks[tw_id].append(chunk_idx)

    # reply edges: if tweet i replies to j, connect chunk i's to chunk j's
    for t_i, t_j in reply_map.items():
        if t_j == -1:
            continue
        # all chunks from t_i get edges to all chunks from t_j
        for ci in tweet_to_chunks[t_i]:
            for cj in tweet_to_chunks[t_j]:
                rows.append(ci)
                cols.append(cj)
                data.append(w_reply)
                # maybe reversed edge w/ smaller weight?
                rows.append(cj)
                cols.append(ci)
                data.append(w_reply * 0.5)

    # same-user adjacency
    # user_map[u] is a list of tweet_ids from that user
    for u, tweet_ids in user_map.items():
        tweet_ids = list(tweet_ids)
        for i in range(len(tweet_ids)):
            for j in range(i + 1, len(tweet_ids)):
                t_i, t_j = tweet_ids[i], tweet_ids[j]
                # connect all chunk combos
                for ci in tweet_to_chunks[t_i]:
                    for cj in tweet_to_chunks[t_j]:
                        rows.append(ci)
                        cols.append(cj)
                        data.append(w_user)
                        # symmetrical
                        rows.append(cj)
                        cols.append(ci)
                        data.append(w_user)

    a_struct = sp.coo_matrix((data, (rows, cols)), shape=(n_chunks, n_chunks)).tocsr()

    # normalize columns (like your build_adj_matrix does)
    a_struct_t = a_struct.transpose().tocsr()
    row_sums = np.array(a_struct_t.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1e-9
    for i in range(a_struct_t.shape[0]):
        start, end = a_struct_t.indptr[i], a_struct_t.indptr[i + 1]
        a_struct_t.data[start:end] /= row_sums[i]

    return a_struct_t.transpose().tocsr()


class SparseRAG:
    def __init__(
        self,
        chunks,
        tweet_index_for_chunk=None,
        reply_map=None,
        user_map=None,
        w_text=0.7,
        w_struct=0.3,
    ):
        """
        pass tweet_index_for_chunk, reply_map, user_map if you want structural adjacency.
        set w_text=1.0, w_struct=0.0 if you only want text adjacency, etc.
        """
        self.chunks = chunks
        self.w_text = w_text
        self.w_struct = w_struct

        t0 = time.time()
        self.embeddings, self.vectorizer = self.build_sparse_embeddings(chunks)
        print(f"Building sparse embeddings: {time.time() - t0:.3f}s")

        t0 = time.time()
        self.text_adj_matrix = self.build_adj_matrix(self.embeddings)
        print(f"Building adjacency matrix: {time.time() - t0:.3f}s")

        self.struct_adj_matrix = None
        if (
            tweet_index_for_chunk is not None
            and reply_map is not None
            and user_map is not None
        ):
            self.struct_adj_matrix = build_structural_adjacency(
                tweet_index_for_chunk,
                reply_map,
                user_map,
                self.text_adj_matrix.shape[0],
            )
            # combine them
            self.adj_matrix = self.mix_adj_matrices(
                self.text_adj_matrix, self.struct_adj_matrix, w_text, w_struct
            )
        else:
            self.adj_matrix = self.text_adj_matrix

    def build_sparse_embeddings(self, chunks: list[str]):
        vectorizer = TfidfVectorizer(dtype=np.float32)
        embeddings = vectorizer.fit_transform(chunks)
        return sp.csr_matrix(embeddings), vectorizer

    def build_adj_matrix(
        self, embeddings: sp.csr_matrix, max_edges=50_000_000
    ) -> sp.csr_matrix:
        sim = embeddings.dot(embeddings.T).tocoo()
        if len(sim.data) > max_edges:
            threshold = np.partition(sim.data, -max_edges)[-max_edges]
            print(f"Pruning edges below threshold: {threshold}")
        else:
            threshold = 0.0
        mask = sim.data >= threshold
        sim = sp.coo_matrix(
            (sim.data[mask], (sim.row[mask], sim.col[mask])), shape=sim.shape
        ).tocsr()

        # normalize columns
        sim_t = sim.transpose().tocsr()
        row_sums = np.array(sim_t.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1e-9
        for i in range(sim_t.shape[0]):
            start, end = sim_t.indptr[i], sim_t.indptr[i + 1]
            sim_t.data[start:end] /= row_sums[i]
        return sim_t.transpose().tocsr()

    def mix_adj_matrices(self, a_text, a_struct, w_text, w_struct):
        a_mixed = a_text.multiply(w_text) + a_struct.multiply(w_struct)
        # re-normalize columns
        a_mixed_t = a_mixed.transpose().tocsr()
        row_sums = np.array(a_mixed_t.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1e-9
        for i in range(a_mixed_t.shape[0]):
            start, end = a_mixed_t.indptr[i], a_mixed_t.indptr[i + 1]
            a_mixed_t.data[start:end] /= row_sums[i]
        return a_mixed_t.transpose().tocsr()

    def personalized_pagerank(
        self, a: sp.csr_matrix, p: sp.csr_matrix, alpha, max_iter=18, tol=1e-6
    ):
        n = a.shape[0]
        pr = np.ones((n, 1), dtype=np.float32) / n
        for _ in range(max_iter):
            old = pr
            pr = (1 - alpha) * a.dot(pr) + alpha * p
            if np.linalg.norm(pr - old, 1) < tol:
                break
        return pr.ravel()

    def get_query_sims(self, query_text):
        query_vec = self.vectorizer.transform([query_text])
        query_sims = sp.csr_matrix.dot(self.embeddings, query_vec.T)
        query_sum = query_sims.sum()
        if query_sum == 0:
            query_sum = 1e-9
        query_sims.data /= query_sum
        return query_sims

    def query(self, query_text, alpha_local=0.6, k=100):
        query_sims = self.get_query_sims(query_text)
        pr_scores = self.personalized_pagerank(self.adj_matrix, query_sims, alpha_local)
        topk = np.argsort(-pr_scores)[:k]
        top_chunks = [self.chunks[i] for i in topk]
        return "\n".join(top_chunks), pr_scores
