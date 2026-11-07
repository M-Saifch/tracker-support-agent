import os
import json
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-120b"

# ---------- RAG setup (step5 wala) ----------
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def chunk_text(text, chunk_size, overlap):
    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step)]

with open("tracker_faq.txt", "r", encoding="utf-8") as f:
    chunks = chunk_text(f.read(), chunk_size=40, overlap=10)

chroma = chromadb.Client()
collection = chroma.create_collection(name="tracker_faq", embedding_function=ef)
collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=[{"source": "tracker_faq.txt", "chunk_no": i} for i in range(len(chunks))],
)

# ---------- Tool 1: FAQ search ----------
def search_faq(question):
    results = collection.query(query_texts=[question], n_results=2)
    return {"faq_chunks": results["documents"][0]}

# ---------- Tool 2: live device data (mock) ----------
DEVICES = {
    "123": {"battery": 82, "status": "moving"},
    "456": {"battery": 15, "status": "idle"},
}

def get_battery_status(device_id):
    device = DEVICES.get(device_id)
    if device is None:
        return {"error": f"Device {device_id} not found"}
    return {
        "device_id": device_id,
        "battery_percent": device["battery"],
        "status": device["status"],
    }

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_faq",
            "description": "Search the tracker FAQ for general product questions: charging, battery life, waterproofing, location update interval, setup, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The user's question, in English"}
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_battery_status",
            "description": "Get the LIVE battery percentage and status of one specific tracker by its device ID. Use only when the user gives a device ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string", "description": "The tracker's device ID, e.g. '123'"}
                },
                "required": ["device_id"],
            },
        },
    },
]

available_functions = {
    "search_faq": search_faq,
    "get_battery_status": get_battery_status,
}

SYSTEM_PROMPT = (
    "You are a support assistant for a GPS tracker company. "
    "Rules: "
    "1) For general product questions, call search_faq and answer ONLY from the returned chunks. "
    "If the chunks do not contain the answer, say: \"I don't know based on the FAQ.\" "
    "2) For live data of a specific device, call get_battery_status. "
    "3) If the question is not about our GPS trackers, politely say you can only help with tracker support. "
    "Do not answer it from your own knowledge."
)

def run_agent(question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(5):  # max 5 rounds
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, tool_choice="auto"
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content  # final jawab

        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            print(f"[TOOL CALL] {call.function.name}({args})")
            result = available_functions[call.function.name](**args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                }
            )

    return "Sorry, I could not finish this request."

questions = [
    "How long does a full charge take?",
    "What is the battery level of device 123?",
    "How long does charging take, and what is the battery of device 456?",
    "What is the capital of France?",
    "Does the tracker support 5G?",
]

for q in questions:
    print(f"\nQ: {q}")
    print(run_agent(q))
    print("-----")