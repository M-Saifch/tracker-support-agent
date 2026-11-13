import streamlit as st
from agent import build_collection, new_history, run_agent, add_document

st.set_page_config(page_title="Tracker Support Agent")
st.title("Tracker Support Agent")


@st.cache_resource
def load_collection():
    return build_collection()


with st.spinner("Loading FAQ..."):
    collection = load_collection()

# created only on the first run; survives reruns
if "history" not in st.session_state:
    st.session_state.history = new_history()
    st.session_state.chat_log = []
    st.session_state.uploaded = set()  # names of files already added

# sidebar: upload a document
with st.sidebar:
    st.header("Add a document")
    file = st.file_uploader("Upload a .txt file", type=["txt"])
    if file is not None and file.name not in st.session_state.uploaded:
        text = file.getvalue().decode("utf-8", errors="ignore")
        n = add_document(collection, text, file.name)
        st.session_state.uploaded.add(file.name)
        st.success(f"{file.name} added ({n} chunks)")
    if st.session_state.uploaded:
        st.caption("Added: " + ", ".join(sorted(st.session_state.uploaded)))

# show the previous chat
for m in st.session_state.chat_log:
    with st.chat_message(m["role"]):
        st.write(m["content"])

prompt = st.chat_input("Ask about your tracker...")
if prompt:
    st.session_state.chat_log.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = run_agent(st.session_state.history, prompt, collection)
            except Exception as e:
                answer = f"Sorry, something went wrong ({type(e).__name__}). Please try again."
        st.write(answer)

    st.session_state.chat_log.append({"role": "assistant", "content": answer})