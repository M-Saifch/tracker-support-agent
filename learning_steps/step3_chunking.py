import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.Client()
collection = client.create_collection(
    name="tracker_docs",
    embedding_function=ef,
)

# baqi code (chunks, collection.add, collection.query) pehle jaisa hi

chunks = [
    "The GPS tracker has a 7 day battery life.",
    "To charge the device, use the USB-C cable that comes in the box.",
    "The tracker sends its location every 30 seconds when moving.",
    "The device is waterproof up to 1 meter for 30 minutes.",
    "If the tracker loses signal, it stores data and uploads it later.",
]

collection.add(
    documents = chunks,
    ids=["c0", "c1", "c2", "c3", "c4"],

)

results = collection.query(
    query_texts = ["How do i charge my tracker"],
    n_results = 2 ,
)

print(results["documents"])
print(results["distances"])