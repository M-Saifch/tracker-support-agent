import chromadb
from chromadb.utils import embedding_functions

def chunk_text(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
    return chunks

with open ("tracker_faq.txt" , "r" , encoding="utf-8") as f:
    text = f.read()

chunks = chunk_text(text, chunk_size=40, overlap=8)
print(f"Total chunks: {len(chunks)}")

ids = [f"chunk_{i}" for i in range(len(chunks))]
metadatas = [{"source": "tracker_faq.txt", "chunk_no": i} for i in range(len(chunks))]

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


client = chromadb.Client()
collection = client.create_collection(name="faq", embedding_function=ef)

collection.add(documents=chunks, ids=ids, metadatas=metadatas)

results = collection.query(query_texts=["How long does charging take?"], n_results=2)

for doc, meta, dist in zip(
    results["documents"][0], results["metadatas"][0], results["distances"][0]
):
    print(f"[{meta['source']} | chunk {meta['chunk_no']}] distance={dist:.3f}")
    print(doc)
    print()