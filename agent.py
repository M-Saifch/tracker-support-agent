import os
import json
import time
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-120b"
MAX_TURNS = 6

SYSTEM_PROMPT = (
    "You are a support assistant for a GPS tracker company. "
    "Rules: "
    "1) For EVERY question about our trackers, including follow-up questions, "
    "call search_faq first, even if you think the answer is already in the conversation. "
    "Answer ONLY from the chunks returned by that latest search. "
    "2) Every sentence in your answer must be directly stated in the returned chunks. "
    "Do not add warnings, examples, side effects, or assumptions, even if they seem obviously true. "
    "Stay close to the FAQ's own wording. Keep answers short. "
    "If the chunks do not contain the answer, say: \"I don't know based on the FAQ.\" "
    "3) For live data of a specific device, call get_battery_status. "
    "4) If the question is not about our GPS trackers, politely say you can only help with tracker support. "
    "Do not answer it from your own knowledge."
)


# ---------- RAG setup ----------
def chunk_text(text, chunk_size, overlap):
    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), step)]

def add_document(collection, text, filename):
    chunks = chunk_text(text, chunk_size=40, overlap=10)
    if not chunks:
        return 0
    collection.upsert(
        documents=chunks,
        ids=[f"{filename}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": filename, "chunk_no": i} for i in range(len(chunks))],
    )
    return len(chunks)


def build_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    faq_path = os.path.join(os.path.dirname(__file__), "tracker_faq.txt")
    with open(faq_path, "r", encoding="utf-8") as f:
        chunks = chunk_text(f.read(), chunk_size=40, overlap=10)

    chroma = chromadb.Client()
    collection = chroma.get_or_create_collection(name="tracker_faq", embedding_function=ef)
    collection.upsert(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": "tracker_faq.txt", "chunk_no": i} for i in range(len(chunks))],
    )
    return collection


# ---------- Tools ----------
def search_faq(question, collection):
    results = collection.query(query_texts=[question], n_results=2)
    return {"faq_chunks": results["documents"][0]}


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
            "description": "Search the tracker FAQ and any uploaded tracker documents for general product questions: charging, battery life, waterproofing, location update interval, setup, etc",
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


def call_tool(name, args, collection):
    if name == "search_faq":
        return search_faq(args["question"], collection)
    if name == "get_battery_status":
        return get_battery_status(args["device_id"])
    return {"error": f"Unknown tool {name}"}


# ---------- Memory ----------
def new_history():
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def get_role(m):
    return m["role"] if isinstance(m, dict) else m.role


def trim_history(history):
    user_idx = [i for i, m in enumerate(history) if get_role(m) == "user"]
    if len(user_idx) > MAX_TURNS:
        cut = user_idx[-MAX_TURNS]
        history[1:] = history[cut:]  # system prompt (index 0) is kept


# ---------- LLM call with retry ----------
def call_llm(messages, max_retries=4):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL, messages=messages, tools=tools,
                tool_choice="auto", temperature=0,
            )
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"[RETRY {attempt + 1}] {type(e).__name__}, retrying in {wait}s...")
            time.sleep(wait)


# ---------- Agent loop ----------
def run_agent(history, question, collection):
    start = len(history)
    history.append({"role": "user", "content": question})
    try:
        for _ in range(5):
            response = call_llm(history)
            msg = response.choices[0].message

            if not msg.tool_calls:
                history.append({"role": "assistant", "content": msg.content})
                trim_history(history)
                return msg.content

            history.append(msg)
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments)
                print(f"[TOOL CALL] {call.function.name}({args})")
                result = call_tool(call.function.name, args, collection)
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result),
                    }
                )
        del history[start:]
        return "Sorry, I could not finish this request."
    except Exception:
        del history[start:]  # remove the unfinished question from history
        raise