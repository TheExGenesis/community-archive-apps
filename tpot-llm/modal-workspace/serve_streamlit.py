import modal
from pathlib import Path

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "streamlit~=1.40.0",
    "numpy",
    "pandas",
    "sentence-transformers",
    "torch",
    "transformers",
    "pyarrow",
    "openai",
)

# Add local files to the image
ui_path = Path(__file__).parent / "ui.py"
image = image.add_local_file(ui_path, remote_path="/root/ui.py")
image = image.add_local_dir(Path(__file__).parent / "utils", remote_path="/root/utils")

app = modal.App("text-rag-ui")

# Volumes for persistent storage
embeddings_volume = modal.Volume.from_name("tpot-llm", create_if_missing=True)
cache_volume = modal.Volume.from_name("tpot-llm-cache", create_if_missing=True)


@app.function(
    image=image,
    # gpu=modal.gpu.T4(),
    volumes={"/tpot-llm/": embeddings_volume, "/cache/": cache_volume},
    secrets=[modal.Secret.from_name("openrouter-api-key")],
    allow_concurrent_inputs=100,
)
@modal.web_server(8000)
def run():
    import shlex
    import subprocess

    target = shlex.quote(str("/root/ui.py"))
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)
