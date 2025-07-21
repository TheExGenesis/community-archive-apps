import sys
from pathlib import Path
import logging
from datetime import datetime
import time
import numpy as np
import modal
import json
import pickle
from typing import Optional
import math

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Config
GPU_CONFIG = "T4"
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 32
MB_PER_ITEM = 0.5
BASE_MEMORY = 128
MAX_MEMORY = 32768


def calculate_memory_requirement(n_items: int) -> int:
    return min(BASE_MEMORY + int(n_items * MB_PER_ITEM), MAX_MEMORY)


def estimate_clustering_params(n_texts: int) -> dict:
    """Auto-estimate clustering parameters based on dataset size"""
    if n_texts < 10:
        return {
            "min_cluster_size": 2,
            "min_samples": 1,
            "n_components": min(2, n_texts - 1),
            "n_neighbors": min(3, max(2, n_texts - 1)),
            "min_dist": 0.0,
        }
    elif n_texts < 50:
        return {
            "min_cluster_size": max(2, int(n_texts * 0.05)),
            "min_samples": max(1, int(n_texts * 0.02)),
            "n_components": min(5, max(2, int(math.log10(n_texts) * 2))),
            "n_neighbors": min(15, max(3, int(math.sqrt(n_texts)))),
            "min_dist": 0.1,
        }
    elif n_texts < 500:
        return {
            "min_cluster_size": max(5, int(n_texts * 0.02)),
            "min_samples": max(2, int(n_texts * 0.01)),
            "n_components": min(10, max(5, int(math.log10(n_texts) * 2))),
            "n_neighbors": min(30, max(10, int(math.sqrt(n_texts)))),
            "min_dist": 0.1,
        }
    elif n_texts < 5000:
        return {
            "min_cluster_size": max(10, int(n_texts * 0.01)),
            "min_samples": max(5, int(n_texts * 0.005)),
            "n_components": min(20, max(10, int(math.log10(n_texts) * 3))),
            "n_neighbors": min(50, max(15, int(math.sqrt(n_texts)))),
            "min_dist": 0.1,
        }
    else:
        return {
            "min_cluster_size": max(20, int(n_texts * 0.005)),
            "min_samples": max(10, int(n_texts * 0.002)),
            "n_components": min(50, max(20, int(math.log10(n_texts) * 4))),
            "n_neighbors": min(100, max(30, int(math.sqrt(n_texts)))),
            "min_dist": 0.05,
        }


def save_results(results: dict, filepath: str, format: str = "json") -> None:
    """Save results to file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj

        with open(filepath.with_suffix(".json"), "w") as f:
            json.dump(convert_numpy(results), f, indent=2)
    else:
        with open(filepath.with_suffix(".pkl"), "wb") as f:
            pickle.dump(results, f)


def load_results(filepath: str) -> dict:
    """Load results from file"""
    filepath = Path(filepath)
    with open(filepath, "rb" if filepath.suffix == ".pkl" else "r") as f:
        return pickle.load(f) if filepath.suffix == ".pkl" else json.load(f)


app = modal.App(name="embedding-clustering-service")

infinity_image = (
    modal.Image.from_registry("michaelf34/infinity:0.0.63")
    .pip_install(
        "infinity-emb[all]",
        "httpx",
        "numpy~=1.26.4",
        "xformers",
        "transformers>=4.53.1",
        "torch>=2.0.0",
    )
    .entrypoint([])
)

rapids_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/rapidsai/base:24.10-cuda12.0-py3.12", add_python="3.12"
    )
    .apt_install("build-essential")
    .pip_install("numpy~=1.26.4", "pandas~=2.2.2", "scikit-learn", "hdbscan")
    .entrypoint([])
)

with infinity_image.imports():
    from infinity_emb import AsyncEngineArray, EngineArgs
    import httpx

with rapids_image.imports():
    import cuml
    import pandas as pd
    import numpy as np


@app.cls(
    gpu=GPU_CONFIG,
    image=infinity_image,
    max_containers=10,
    memory=lambda args: calculate_memory_requirement(len(args[0])),
)
@modal.concurrent(max_inputs=10)
class InfinityEmbedder:
    model_name: str = modal.parameter(default=MODEL_ID)
    batch_size: int = modal.parameter(default=BATCH_SIZE)

    @modal.method()
    async def embed(self, texts: list[str]) -> list[list[float]]:
        start = time.time()
        engine = self.engine_array[0]
        embeddings, _ = await engine.embed(sentences=texts)
        logging.info(f"Embedded {len(texts)} texts in {time.time()-start:.3f}s")
        return embeddings

    @modal.enter()
    async def enter(self):
        self.engine_array = AsyncEngineArray.from_args(
            [
                EngineArgs(
                    model_name_or_path=self.model_name,
                    batch_size=self.batch_size,
                    dtype="float16",
                )
            ]
        )
        await self.engine_array.astart()

    @modal.exit()
    async def exit(self):
        if hasattr(self, "engine_array"):
            await self.engine_array.astop()


@app.function(gpu="T4", image=rapids_image, timeout=600)
def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 5,
    n_neighbors: int = 3,
    min_dist: float = 0.0,
) -> tuple[np.ndarray, float]:
    start_time = time.time()
    reducer = cuml.manifold.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
    )
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings, time.time() - start_time


@app.function(
    gpu="T4",
    image=rapids_image,
    timeout=3600,
    memory=lambda args: calculate_memory_requirement(
        len(args[0]) if isinstance(args[0], np.ndarray) else BASE_MEMORY
    ),
)
def cluster_embeddings(
    reduced_embeddings: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    cluster_selection_method: str = "eom",
) -> dict:
    start_time = time.time()
    clusterer = cuml.cluster.hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        metric="euclidean",
        prediction_data=True,
    )
    clusterer.fit(reduced_embeddings)

    n_clusters = len(np.unique(clusterer.labels_[clusterer.labels_ != -1]))
    n_noise = sum(clusterer.labels_ == -1)

    return {
        "labels": [str(l) for l in clusterer.labels_],
        "probabilities": clusterer.probabilities_.tolist(),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "timing": time.time() - start_time,
        "params": {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_method": cluster_selection_method,
        },
    }


@app.function(
    image=infinity_image,
    timeout=3600,
    memory=lambda args: calculate_memory_requirement(len(args[0])),
)
def embed_and_cluster_pipeline(
    texts: list[str],
    auto_params: bool = True,
    n_components: int = None,
    min_cluster_size: int = None,
    min_samples: int = None,
    n_neighbors: int = None,
    min_dist: float = None,
    cluster_selection_method: str = "eom",
    save_to_file: Optional[str] = None,
    save_format: str = "json",
) -> dict:
    """Complete embedding and clustering pipeline"""
    n_texts = len(texts)
    logging.info(f"Starting pipeline for {n_texts} texts")

    # Parameter estimation
    if auto_params:
        estimated = estimate_clustering_params(n_texts)
        final_params = {
            "n_components": n_components or estimated["n_components"],
            "min_cluster_size": min_cluster_size or estimated["min_cluster_size"],
            "min_samples": min_samples or estimated["min_samples"],
            "n_neighbors": n_neighbors or estimated["n_neighbors"],
            "min_dist": min_dist if min_dist is not None else estimated["min_dist"],
        }
    else:
        final_params = {
            "n_components": n_components or 5,
            "min_cluster_size": min_cluster_size or 10,
            "min_samples": min_samples or 5,
            "n_neighbors": n_neighbors or 15,
            "min_dist": min_dist if min_dist is not None else 0.1,
        }

    # Pipeline
    embedder = InfinityEmbedder()
    embeddings = embedder.embed.remote(texts)

    reduced_embeddings, reduction_time = reduce_dimensions.remote(
        np.array(embeddings),
        n_components=final_params["n_components"],
        n_neighbors=final_params["n_neighbors"],
        min_dist=final_params["min_dist"],
    )

    cluster_results = cluster_embeddings.remote(
        reduced_embeddings,
        min_cluster_size=final_params["min_cluster_size"],
        min_samples=final_params["min_samples"],
        cluster_selection_method=cluster_selection_method,
    )

    results = {
        "embeddings": embeddings,
        "reduced_embeddings": reduced_embeddings.tolist(),
        "clustering": cluster_results,
        "reduction_time": reduction_time,
        "n_texts": n_texts,
        "parameters_used": final_params,
        "auto_params": auto_params,
        "texts": texts,
        "timestamp": datetime.now().isoformat(),
    }

    if save_to_file:
        save_results(results, save_to_file, save_format)

    return results


@app.function(
    image=infinity_image,
    timeout=3600,
    memory=lambda args: calculate_memory_requirement(len(args[0])),
)
def embed_texts_only(texts: list[str]) -> list[list[float]]:
    """Embedding-only function for efficient batch processing"""
    embedder = InfinityEmbedder()
    return embedder.embed.remote(texts)


@app.function(
    image=infinity_image,
    timeout=3600,
    memory=lambda args: calculate_memory_requirement(len(args[0])),
)
def cluster_from_embeddings_remote(
    embeddings: list[list[float]], texts: list[str], auto_params: bool = True, **kwargs
) -> dict:
    """Cluster pre-computed embeddings"""
    return embed_and_cluster_pipeline(texts, auto_params, **kwargs)


def analyze_results(results: dict) -> None:
    """Print clustering analysis"""
    texts = results["texts"]
    labels = results["clustering"]["labels"]
    n_clusters = results["clustering"]["n_clusters"]
    n_noise = results["clustering"]["n_noise"]

    print(f"\n=== RESULTS ===")
    print(f"Dataset: {len(texts)} texts | Clusters: {n_clusters} | Noise: {n_noise}")
    print(f"Parameters: {results['parameters_used']}")

    # Group by cluster
    clusters = {}
    for i, (text, label) in enumerate(zip(texts, labels)):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((i, text))

    for cluster_id in sorted(clusters.keys()):
        name = "NOISE" if cluster_id == "-1" else f"CLUSTER {cluster_id}"
        print(f"\n{name} ({len(clusters[cluster_id])} texts):")
        for idx, text in clusters[cluster_id]:
            print(f"  [{idx}] {text[:60]}{'...' if len(text) > 60 else ''}")


@app.local_entrypoint()
def main():
    texts = [
        "The weather is nice today",
        "I love machine learning",
        "Python is a great programming language",
        "The sun is shining brightly",
        "Deep learning models are powerful",
        "JavaScript is used for web development",
        "Data science combines statistics and programming",
        "Neural networks mimic brain functions",
    ]

    print("=== Small Dataset Test ===")
    results = embed_and_cluster_pipeline.remote(texts[:6])
    analyze_results(results)

    print(f"\n✅ Service ready! Found {results['clustering']['n_clusters']} clusters")
