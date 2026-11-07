import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ---------- Setup (step4 wala hissa) ----------
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def chunk_text(text, chunk_size, overlap):
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

with open("tracker_faq.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = chunk_text(text, chunk_size=40, overlap=10)

client = chromadb.Client()
collection = client.create_collection(name="tracker_faq", embedding_function=ef)
collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=[{"source": "tracker_faq.txt", "chunk_no": i} for i in range(len(chunks))],
)

# ---------- RAG (naya hissa) ----------
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

def ask(question, n_results=2):
    results = collection.query(query_texts=[question], n_results=n_results)
    context = "\n\n".join(results["documents"][0])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a support assistant for a GPS tracker company. "
                "Answer ONLY using the context provided. "
                "If the answer is not in the context, say: "
                "\"I don't know based on the FAQ.\""
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

print(ask("How long does a full charge take?"))
print(ask("Is the tracker waterproof?"))
print(ask("What is the capital of France?"))
print(ask("Does the tracker support 5G?"))