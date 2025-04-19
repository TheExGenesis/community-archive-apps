"""
RAG on tweets

DONE - put /borg-ca-tpot.parquet into modal volume
TODO - embed them, save them to modal volume
TODO - serve them so we can query them
TODO - rerank?
TODO - serve tweet generation with RAG results

"""

# %%
import modal

assert modal.__version__ > "0.49.0"
modal.__version__
from modal import app

app = modal.App(name="example-basic-notebook-app")

GPU_CONFIG = modal.gpu.T4()
MODEL_ID = "Alibaba-NLP/gte-Qwen2-7B-instruct"
MODEL_ID = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 32
DOCKER_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:turing-1.5"  # Turing for T4s  # Create the app before using it in decorators
EMBEDDINGS_PATH = "/tpot-llm/borg-ca-tpot-embeddings.npy"

import subprocess
import asyncio
import socket


def download_model():
    spawn_server().terminate()


# Create TEI image with all necessary dependencies
tei_image = (
    modal.Image.from_registry(DOCKER_IMAGE, add_python="3.10")
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install(
        "httpx",
        "numpy~=1.26.4",
        "pandas~=2.2.2",
        "supabase",
        "tqdm",
        "seaborn",
        "openai",
        "toolz",
        "pyarrow",
    )
    .run_function(download_model, gpu=GPU_CONFIG)
)

with tei_image.imports():
    import httpx


# Add TEI server setup functions
def spawn_server() -> subprocess.Popen:
    process = subprocess.Popen(
        [
            "text-embeddings-router",
            "--model-id",
            MODEL_ID,
            "--port",
            "8000",
            "--max-client-batch-size",
            "128",
            "--dtype",
            "float16",
            "--auto-truncate",
        ]
    )

    while True:
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            print("Webserver ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(f"launcher exited unexpectedly with code {retcode}")


# Now we can use app.cls since app is defined
@app.cls(
    gpu=GPU_CONFIG,
    image=tei_image,
    concurrency_limit=10,
    allow_concurrent_inputs=10,
)
class TextEmbeddingsInference:
    @modal.enter()
    def setup_server(self):
        self.process = spawn_server()
        self.client = httpx.AsyncClient(base_url="http://127.0.0.1:8000")

    @modal.exit()
    def teardown_server(self):
        self.process.terminate()

    @modal.method()
    async def embed(self, inputs: list[str]):
        retries = 3  # Number of retries
        for attempt in range(retries):
            try:
                resp = await self.client.post("/embed", json={"inputs": inputs})
                resp.raise_for_status()
                return resp.json()
            except httpx.ReadTimeout as e:
                if attempt < retries - 1:  # If not the last attempt
                    print(f"Timeout occurred, retrying... (Attempt {attempt + 1})")
                    await asyncio.sleep(1)  # Wait before retrying
                else:
                    raise e  # Raise the last exception if all retries fail


# load from modal volume
volume = modal.Volume.lookup("tpot-llm")
import pandas as pd


import numpy as np


import io


@app.function(
    gpu=GPU_CONFIG,
    image=tei_image,
    volumes={"/tpot-llm": volume},
    timeout=36000,
)
def embed_dataset(batch_size: int = 32, batches_per_chunk: int = 10000):
    # Load data
    df = pd.read_parquet("/tpot-llm/borg-ca-tpot.parquet")
    texts = df.processed_text.tolist()

    # Load existing checkpoints
    embeddings = []

    embedder = TextEmbeddingsInference()

    # Split into batches of 32
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    embeddings = []
    batch_embeddings = embedder.embed.map(batches, order_outputs=True)
    for batch_embedding in batch_embeddings:
        embeddings.extend(batch_embedding)

    # Convert to numpy array
    embeddings = np.array(embeddings)

    # Concatenate all embeddings along first dimension
    np.save(EMBEDDINGS_PATH, embeddings)


@app.local_entrypoint()
def main():
    embed_dataset.remote()
