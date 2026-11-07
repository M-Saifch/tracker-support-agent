import os
import json
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
import time 

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

MAX_TURNS = 6  # kitne purane sawal yaad rakhne hain

# Memory: system prompt hamesha pehla message rahega
history = [{"role": "system", "content": SYSTEM_PROMPT}]

def get_role(m):
    # messages kabhi dict hote hain, kabhi Groq ke objects
    return m["role"] if isinstance(m, dict) else m.role

def trim_history():
    user_idx = [i for i, m in enumerate(history) if get_role(m) == "user"]
    if len(user_idx) > MAX_TURNS:
        cut = user_idx[-MAX_TURNS]
        history[1:] = history[cut:]  # system prompt (index 0) bacha rahega

def call_llm(messages, max_retries=4):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL, messages=messages, tools=tools, tool_choice="auto" , temperature=0
            )
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
            if attempt == max_retries - 1:
                raise  # aakhri koshish bhi fail, error upar bhej do
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"[RETRY {attempt + 1}] {type(e).__name__}, {wait}s baad dobara...")
            time.sleep(wait)

def run_agent(question):
    start = len(history)
    history.append({"role": "user", "content": question})
    try:
        for _ in range(5):
            response = call_llm(history)
            msg = response.choices[0].message

            if not msg.tool_calls:
                history.append({"role": "assistant", "content": msg.content})
                trim_history()
                return msg.content

            history.append(msg)
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments)
                print(f"[TOOL CALL] {call.function.name}({args})")
                result = available_functions[call.function.name](**args)
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
        del history[start:]  # adhoora sawal history se hata do
        raise

# Chat loop
print("Tracker Support Bot (band karne ke liye 'exit' likho)")
while True:
    q = input("\nYou: ").strip()
    if q.lower() == "exit":
        break
    if not q:
        continue
    try:
        print("Bot:", run_agent(q))
    except Exception as e:
        print(f"Bot: Sorry, abhi masla aa gaya ({type(e).__name__}). Dobara koshish karo.")