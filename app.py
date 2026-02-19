import os
import streamlit as st
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime
from pypdf import PdfReader
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import tempfile
import subprocess
import asyncio
import edge_tts

# -------------------- LOAD ENV --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# -------------------- EMBEDDING MODEL --------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(page_title="AgriSahai AI 🌾", page_icon="🌱", layout="wide")

# -------------------- SIDEBAR --------------------
st.sidebar.title("🌾 AgriSahai Controls")

language = st.sidebar.selectbox("🌍 Select Language", ["English", "Telugu", "Hindi"])

model_choice = st.sidebar.selectbox(
    "🤖 Groq Model",
    ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"]
)

temperature = st.sidebar.slider("🎭 Response Creativity", 0.0, 1.0, 0.4)
top_k = st.sidebar.slider("📌 FAISS Context Docs", 1, 5, 2)

voice_output = st.sidebar.checkbox("🔊 Enable Voice Output", value=False)

st.sidebar.markdown("---")

# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Save Chat
if st.sidebar.button("📥 Save Chat to CSV"):
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        chat_data = []
        for msg in st.session_state.messages:
            chat_data.append({
                "role": msg["role"],
                "message": msg["content"],
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        df = pd.DataFrame(chat_data)
        df.to_csv("kisansahai_chat.csv", index=False)
        st.sidebar.success("✅ Chat saved as kisansahai_chat.csv")
    else:
        st.sidebar.warning("⚠️ No chat available to save.")

st.sidebar.markdown("---")
st.sidebar.info("📌 Upload PDF below to expand agriculture knowledge.")

uploaded_pdf = st.sidebar.file_uploader("📄 Upload Agriculture PDF", type=["pdf"])

# -------------------- DEFAULT KNOWLEDGE BASE --------------------
default_documents = [
    "For wheat crop, irrigation should be done every 20 to 25 days depending on soil moisture.",
    "If tomato leaves are curling, it may be due to viral infection or overwatering.",
    "To control aphids in cotton, use neem oil spray or recommended insecticides.",
    "DAP fertilizer is usually applied during sowing stage for better root growth.",
    "Yellow leaves in rice may indicate nitrogen deficiency.",
    "If pests are attacking chilli plants, use neem-based pesticides or consult agricultural officer for recommended spray.",
    "For improving soil fertility, farmers should use organic compost and rotate crops regularly.",
    "If sugarcane crop shows red rot disease symptoms, remove infected plants and use resistant varieties.",
    "If brinjal has fruit borer problem, use pheromone traps and recommended pesticide spray."
]

documents = default_documents.copy()

# -------------------- PDF FUNCTIONS --------------------
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def chunk_text(text, chunk_size=400):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


# If PDF uploaded
if uploaded_pdf:
    pdf_text = extract_pdf_text(uploaded_pdf)
    pdf_chunks = chunk_text(pdf_text)

    if len(pdf_chunks) > 0:
        documents.extend(pdf_chunks)
        st.sidebar.success(f"✅ PDF Loaded ({len(pdf_chunks)} chunks added)")
    else:
        st.sidebar.warning("⚠️ PDF has no readable text.")

# -------------------- BUILD FAISS INDEX --------------------
doc_embeddings = embed_model.encode(documents)
dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# -------------------- SPEECH TO TEXT FUNCTION --------------------
def speech_to_text(audio_bytes):
    recognizer = sr.Recognizer()

    # Save audio as webm
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
        temp_webm.write(audio_bytes)
        webm_path = temp_webm.name

    # Convert webm to wav using ffmpeg
    wav_path = webm_path.replace(".webm", ".wav")

    subprocess.run(
        ["ffmpeg", "-y", "-i", webm_path, wav_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Read wav file
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "❌ Could not understand audio"
    except sr.RequestError:
        return "❌ Speech recognition service error"

# -------------------- TEXT TO SPEECH (EDGE TTS) --------------------
async def generate_tts(text, voice):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        file_path = fp.name

    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(file_path)

    return file_path

# -------------------- MAIN UI --------------------
st.title("📞🌾 AgriSahai AI - Farmer Call Centre Assistant")
st.caption("Chatbot + PDF RAG + Voice Input + Voice Output (Edge Neural Voices)")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- VOICE INPUT UI --------------------
st.markdown("### 🎤 Voice Input (Speak your query)")
audio = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="⏹️ Stop Recording", key="recorder")

voice_query = None
if audio:
    if "bytes" in audio:
        voice_query = speech_to_text(audio["bytes"])
        st.success(f"🗣️ You said: {voice_query}")

# -------------------- TEXT INPUT --------------------
user_query = st.chat_input("Type your farming query here...")

# If voice query exists, prefer it
final_query = voice_query if voice_query else user_query

# -------------------- PROCESS QUERY --------------------
if final_query:
    st.session_state.messages.append({"role": "user", "content": final_query})

    with st.chat_message("user"):
        st.markdown(final_query)

    # Embed query
    query_embedding = embed_model.encode([final_query])

    # Search FAISS
    D, I = index.search(np.array(query_embedding), k=top_k)
    retrieved_docs = [documents[i] for i in I[0]]
    context = "\n".join(retrieved_docs)

    prompt = f"""
You are KisanSahai, an agriculture call centre assistant.

Instructions:
- Reply ONLY in {language}.
- Use simple farmer-friendly language.
- Give bullet points.
- Suggest natural methods first.
- If chemical pesticide needed, say "Consult agriculture officer before using chemical spray".
- If serious disease, suggest visiting agriculture office.

Context:
{context}

Farmer Question:
{final_query}

Answer:
"""

    response = client.chat.completions.create(
        model=model_choice,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )

    bot_reply = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    with st.chat_message("assistant"):
        st.markdown(bot_reply)

    # -------------------- VOICE OUTPUT --------------------
    if voice_output:
        if language == "English":
            voice = "en-IN-NeerjaNeural"
        elif language == "Hindi":
            voice = "hi-IN-SwaraNeural"
        else:
            voice = "te-IN-MohanNeural"

        mp3_path = asyncio.run(generate_tts(bot_reply, voice))
        st.audio(mp3_path, format="audio/mp3")

    # Context expander
    with st.expander("📌 Retrieved Context Used"):
        st.write(context)

st.markdown("---")
st.markdown("🌱 **AgriSahai AI** | Farmer Support Call Centre Assistant")
