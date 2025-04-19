# %%
import chromadb

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")
collection.add(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges",
    ],
    ids=["id1", "id2"],
)
results = collection.query(
    query_texts=[
        "This is a query document about hawaii"
    ],  # Chroma will embed this for you
    n_results=2,  # how many results to return
)
print(results)
{
    "documents": [
        ["This is a document about pineapple", "This is a document about oranges"]
    ],
    "ids": [["id1", "id2"]],
    "distances": [[1.0404009819030762, 1.243080496788025]],
    "uris": None,
    "data": None,
    "metadatas": [[None, None]],
    "embeddings": None,
}
# %%
import chromadb

chroma_client = chromadb.Client()

# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(name="my_collection")

# switch `add` to `upsert` to avoid adding the same documents every time
collection.upsert(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges",
    ],
    ids=["id1", "id2"],
)

results = collection.query(
    query_texts=[
        "This is a query document about florida"
    ],  # Chroma will embed this for you
    n_results=2,  # how many results to return
)

print(results)

# %%
