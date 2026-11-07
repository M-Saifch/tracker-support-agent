# Tracker Support Agent

A support chatbot for GPS tracker customers, built with RAG and function calling.
It answers questions from a product FAQ and can call tools (e.g. check device battery status).

## Features
- RAG pipeline: retrieve (ChromaDB + sentence-transformers) → augment → generate (Groq LLM)
- Function calling agent loop with a max-round safety limit (tools: `search_faq`, `get_battery_status`)
- Conversation memory with history trimming
- Retry with exponential backoff on temporary API errors (rate limit, network)
- System-prompt guardrails (refuses out-of-scope questions, says "I don't know" if the FAQ has no answer)

## Setup
1. Clone the repo and create a virtual environment
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your Groq API key
4. Run `python step8_memory.py`

## Project structure
- `step3_chunking.py` / `step4_loadFile.py`: chunking with overlap, loading the FAQ into Chroma
- `step5_RAG.py`: basic RAG
- `step6_toolCalling.py` / `step7_agent.py`: function calling and the agent loop
- `step8_memory.py`: conversation memory + retry logic
- `tracker_faq.txt`: sample FAQ data

## Known limitations
- Memory is session-only (lost when the script stops)
- Occasional minor hallucination on follow-up questions
- `get_battery_status` uses mock data

## Roadmap
Streamlit UI, FastAPI endpoint, evaluation test set