import os
import io
import tempfile
import subprocess
import asyncio
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import edge_tts

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# -------------------- CONFIG --------------------
st.set_page_config(page_title="AgriSahai AI 🌾", page_icon="🌾", layout="wide")


# -------------------- LOAD CSS --------------------
def load_css():
    if os.path.exists("ui/styles.css"):
        with open("ui/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


# -------------------- ENV --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


# -------------------- DEFAULT DOCS --------------------
DEFAULT_DOCS = [
    "For wheat crop, irrigation should be done every 20 to 25 days depending on soil moisture.",
    "If tomato leaves are curling, it may be due to viral infection or overwatering.",
    "To control aphids in cotton, use neem oil spray or recommended insecticides.",
    "DAP fertilizer is usually applied during sowing stage for better root growth.",
    "Yellow leaves in rice may indicate nitrogen deficiency.",
    "If pests are attacking chilli plants, use neem-based pesticides or recommended spray.",
    "For improving soil fertility, farmers should use organic compost and rotate crops regularly.",
    "If sugarcane crop shows red rot disease symptoms, remove infected plants and use resistant varieties.",
    "If brinjal has fruit borer problem, use pheromone traps and recommended pesticide spray."
]


# -------------------- GOVT SCHEMES --------------------
GOVT_SCHEMES = [
    "PM-KISAN provides ₹6000 per year income support to eligible farmers in India.",
    "Pradhan Mantri Fasal Bima Yojana (PMFBY) provides crop insurance against natural calamities and pest attacks.",
    "Soil Health Card Scheme helps farmers test soil and get fertilizer recommendations.",
    "Kisan Credit Card (KCC) provides low-interest loans to farmers for crop cultivation.",
    "PMKSY (Pradhan Mantri Krishi Sinchayee Yojana) supports drip irrigation and water conservation subsidies.",
    "NABARD subsidies support rural agriculture infrastructure and dairy development.",
    "National Horticulture Mission provides support for fruits, vegetables and horticulture crops.",
    "Subsidies are available for solar pump installation under PM-KUSUM scheme.",
    "Organic farming support available under Paramparagat Krishi Vikas Yojana (PKVY).",
    "eNAM (National Agriculture Market) provides better market access and digital selling for farmers."
]


# -------------------- MARKET PRICE DATA --------------------
MARKET_PRICE_DATA = {
    "Rice": {"price": 2300, "unit": "Quintal", "trend": "⬆️ Rising"},
    "Cotton": {"price": 7200, "unit": "Quintal", "trend": "⬇️ Slight Drop"},
    "Chilli": {"price": 14500, "unit": "Quintal", "trend": "⬆️ High Demand"},
    "Maize": {"price": 2100, "unit": "Quintal", "trend": "➡️ Stable"},
    "Tomato": {"price": 1800, "unit": "Quintal", "trend": "⬇️ Falling"},
    "Wheat": {"price": 2500, "unit": "Quintal", "trend": "➡️ Stable"},
    "Sugarcane": {"price": 350, "unit": "Per Ton", "trend": "➡️ Stable"}
}


# -------------------- WEATHER DUMMY DATA --------------------
WEATHER_DATA = {
    "Telangana": {"temp": 33, "rain": 20, "wind": 12, "alert": "Normal"},
    "Andhra Pradesh": {"temp": 34, "rain": 15, "wind": 10, "alert": "Heatwave Risk"},
    "Karnataka": {"temp": 30, "rain": 40, "wind": 15, "alert": "Rain Alert"},
    "Tamil Nadu": {"temp": 35, "rain": 10, "wind": 9, "alert": "High Heat"},
    "Maharashtra": {"temp": 31, "rain": 35, "wind": 18, "alert": "Rain Alert"}
}


# -------------------- CACHE MODEL --------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embed_model = load_embedding_model()


# -------------------- PDF FUNCTIONS --------------------
@st.cache_data(show_spinner=False)
def extract_pdf_text_cached(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def chunk_text(text, chunk_size=350):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


# -------------------- CACHE EMBEDDINGS --------------------
@st.cache_data(show_spinner=False)
def embed_documents_cached(doc_list):
    embeddings = embed_model.encode(doc_list)
    return np.array(embeddings).astype("float32")


@st.cache_resource
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


# -------------------- SPEECH TO TEXT --------------------
def speech_to_text(audio_bytes):
    recognizer = sr.Recognizer()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
        temp_webm.write(audio_bytes)
        webm_path = temp_webm.name

    wav_path = webm_path.replace(".webm", ".wav")

    subprocess.run(
        ["ffmpeg", "-y", "-i", webm_path, wav_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio_data)
    except:
        return None


# -------------------- TEXT TO SPEECH --------------------
async def generate_tts(text, voice):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        file_path = fp.name

    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(file_path)

    return file_path


# -------------------- PDF REPORT GENERATION --------------------
def generate_pdf_report(messages, crop, stage, location):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "AgriSahai AI - Advisory Report")
    y -= 30

    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    y -= 20
    c.drawString(50, y, f"Crop: {crop} | Stage: {stage} | Location: {location}")
    y -= 30

    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Conversation Summary:")
    y -= 20

    c.setFont("Helvetica", 10)

    for msg in messages:
        role = msg["role"].upper()
        text = msg["content"]

        c.drawString(50, y, f"{role}:")
        y -= 15

        for line in text.split("\n"):
            if y < 80:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 10)

            c.drawString(70, y, line[:120])
            y -= 14

        y -= 10

    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, 50, "🌱 AgriSahai AI | Farmer Support Assistant")

    c.save()
    buffer.seek(0)
    return buffer


# -------------------- SESSION INIT --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = DEFAULT_DOCS.copy() + GOVT_SCHEMES.copy()

if "faiss_ready" not in st.session_state:
    st.session_state.faiss_ready = False

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False


# -------------------- BUILD FAISS --------------------
def ensure_faiss_index():
    if not st.session_state.faiss_ready:
        with st.spinner("⚡ Building FAISS Index..."):
            embeddings = embed_documents_cached(st.session_state.documents)
            st.session_state.faiss_index = build_faiss_index(embeddings)
            st.session_state.faiss_ready = True


ensure_faiss_index()


# -------------------- GROQ RESPONSE --------------------
def get_groq_response(query, crop, stage, location, language, model_choice, temperature, top_k):
    query_embedding = embed_model.encode([query]).astype("float32")
    D, I = st.session_state.faiss_index.search(query_embedding, k=top_k)

    retrieved_docs = [st.session_state.documents[i] for i in I[0]]
    context = "\n".join(retrieved_docs)

    prompt = f"""
You are AgriSahai AI, an agriculture officer assistant.

RESPONSE RULES:
- Reply ONLY in {language}
- Use simple easy language
- Use bullet points and step-by-step format
- Organic solution first, chemical solution second
- If chemicals suggested: mention dosage + spray method + precautions
- Keep answers short and actionable
- End with: 🌱 If you want, share crop name + stage + location for more accurate guidance.

Farmer Details:
Crop: {crop}
Stage: {stage}
Location: {location}

Context:
{context}

Farmer Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model=model_choice,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )

    return response.choices[0].message.content


# -------------------- WEATHER PAGE --------------------
def weather_alert_page(language, model_choice):
    st.markdown("<div class='agri-title'>Weather Alerts 🌦️</div>", unsafe_allow_html=True)
    st.markdown("<div class='agri-subtitle'>Weather-based irrigation and crop protection advice</div>", unsafe_allow_html=True)

    st.write("")

    state = st.selectbox("📍 Select State", list(WEATHER_DATA.keys()))
    crop = st.selectbox("🌾 Crop", ["Rice", "Cotton", "Chilli", "Maize", "Tomato", "Wheat", "Sugarcane"])

    weather = WEATHER_DATA[state]

    st.markdown(
        f"<div class='glass-card'><b>🌦 Current Weather</b><br>"
        f"📍 Location: <b>{state}</b><br>"
        f"🌡 Temp: <b>{weather['temp']}°C</b><br>"
        f"🌧 Rain Chance: <b>{weather['rain']}%</b><br>"
        f"💨 Wind: <b>{weather['wind']} km/h</b><br>"
        f"🚨 Alert: <b>{weather['alert']}</b></div>",
        unsafe_allow_html=True
    )

    st.write("")

    if st.button("🧠 Get Weather Advisory"):
        with st.spinner("Generating advisory..."):
            prompt = f"""
You are AgriSahai AI.

Weather Data:
State: {state}
Temperature: {weather['temp']}°C
Rain chance: {weather['rain']}%
Wind speed: {weather['wind']} km/h
Alert: {weather['alert']}

Crop: {crop}

Give weather-based advisory:
- irrigation schedule
- pesticide spray precautions
- crop protection measures
- water saving methods

Rules:
- Reply ONLY in {language}
- Use bullet points
- Keep it short and actionable
- End with: 🌱 If you want, share crop name + stage + location for more accurate guidance.
"""

            response = client.chat.completions.create(
                model=model_choice,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            st.markdown("<div class='glass-card'><b>✅ Weather Advisory</b></div>", unsafe_allow_html=True)
            st.write(response.choices[0].message.content)


# -------------------- GOVT SCHEME PAGE --------------------
def govt_scheme_page(language, model_choice):
    st.markdown("<div class='agri-title'>Govt Scheme Finder 🏛️</div>", unsafe_allow_html=True)
    st.markdown("<div class='agri-subtitle'>Find subsidies and schemes for Indian farmers</div>", unsafe_allow_html=True)

    st.write("")
    state = st.selectbox("📍 Select State", list(WEATHER_DATA.keys()))
    farmer_type = st.selectbox("👨‍🌾 Farmer Type", ["Small Farmer", "Marginal Farmer", "Medium Farmer", "Large Farmer"])
    crop_focus = st.selectbox("🌱 Crop Focus", ["Rice", "Cotton", "Chilli", "Maize", "Tomato", "Any"])

    if st.button("🔍 Find Schemes"):
        with st.spinner("Searching schemes..."):
            prompt = f"""
You are AgriSahai AI.

State: {state}
Farmer Type: {farmer_type}
Crop Focus: {crop_focus}

Suggest top 5 government schemes.
Mention:
- benefit
- eligibility
- documents required
- how to apply

Rules:
- Reply ONLY in {language}
- Use bullet points
- End with: 🌱 If you want, share crop name + stage + location for more accurate guidance.
"""

            response = client.chat.completions.create(
                model=model_choice,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            st.write(response.choices[0].message.content)


# -------------------- MARKET PRICE PAGE --------------------
def market_price_page(language, model_choice):
    st.markdown("<div class='agri-title'>Market Price Advisory 💰</div>", unsafe_allow_html=True)
    st.markdown("<div class='agri-subtitle'>Best time to sell + storage tips + profit guidance</div>", unsafe_allow_html=True)

    crop = st.selectbox("🌾 Select Crop", list(MARKET_PRICE_DATA.keys()))
    state = st.selectbox("📍 Select State", list(WEATHER_DATA.keys()))
    quantity = st.number_input("📦 Quantity (Quintals)", 1.0, 500.0, 10.0)

    base_price = MARKET_PRICE_DATA[crop]["price"]
    unit = MARKET_PRICE_DATA[crop]["unit"]
    trend = MARKET_PRICE_DATA[crop]["trend"]

    st.markdown(
        f"<div class='glass-card'><b>📈 Estimated Price</b><br>"
        f"🌾 Crop: <b>{crop}</b><br>"
        f"💰 Price: <b>₹{base_price} / {unit}</b><br>"
        f"📊 Trend: <b>{trend}</b></div>",
        unsafe_allow_html=True
    )

    if st.button("🧠 Get Selling Advice"):
        with st.spinner("Generating market advisory..."):
            prompt = f"""
You are AgriSahai AI.

State: {state}
Crop: {crop}
Quantity: {quantity} quintals
Price: ₹{base_price} per {unit}
Trend: {trend}

Give:
- best time to sell
- storage tips
- negotiation tips
- mandi vs eNAM advice

Rules:
- Reply ONLY in {language}
- Use bullet points
- End with: 🌱 If you want, share crop name + stage + location for more accurate guidance.
"""

            response = client.chat.completions.create(
                model=model_choice,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            st.write(response.choices[0].message.content)


# -------------------- HEADER --------------------
st.markdown("<div class='agri-title'>AgriSahai AI 🌾</div>", unsafe_allow_html=True)
st.markdown("<div class='agri-subtitle'>Your AI Agriculture Officer in Your Phone</div>", unsafe_allow_html=True)
st.write("")


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## 🚀 Navigation")
    menu = st.radio("Choose Section", ["Crop Advisory Chat", "Weather Alerts", "Govt Schemes", "Market Prices"])

    st.markdown("---")
    language = st.selectbox("🌍 Language", ["English", "Telugu", "Hindi"])
    model_choice = st.selectbox("🤖 Groq Model", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"])
    temperature = st.slider("🎭 Creativity", 0.0, 1.0, 0.4)
    top_k = st.slider("📌 FAISS Docs", 1, 5, 2)
    voice_output = st.checkbox("🔊 Voice Output", value=False)

    st.markdown("---")
    uploaded_pdf = st.file_uploader("📄 Upload Agriculture PDF", type=["pdf"])

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("📥 Save Chat CSV"):
        if len(st.session_state.messages) > 0:
            df = pd.DataFrame(st.session_state.messages)
            df.to_csv("agrisahai_chat.csv", index=False)
            st.success("✅ Saved agrisahai_chat.csv")


# -------------------- PDF PROCESSING --------------------
if uploaded_pdf and not st.session_state.pdf_loaded:
    pdf_bytes = uploaded_pdf.read()
    pdf_text = extract_pdf_text_cached(pdf_bytes)
    pdf_chunks = chunk_text(pdf_text)

    if len(pdf_chunks) > 0:
        st.session_state.documents.extend(pdf_chunks)
        st.session_state.pdf_loaded = True
        st.session_state.faiss_ready = False
        st.success(f"✅ PDF Loaded Successfully ({len(pdf_chunks)} chunks added)")
        st.rerun()


# -------------------- ROUTING --------------------
if menu == "Weather Alerts":
    weather_alert_page(language, model_choice)

elif menu == "Govt Schemes":
    govt_scheme_page(language, model_choice)

elif menu == "Market Prices":
    market_price_page(language, model_choice)

else:
    # -------------------- CHAT PAGE --------------------
    st.markdown("## 🌾 Crop Advisory Chat")
    st.write("")

    col1, col2, col3 = st.columns(3)
    with col1:
        crop = st.selectbox("🌾 Crop", ["Rice", "Cotton", "Chilli", "Maize", "Tomato", "Wheat", "Sugarcane"])
    with col2:
        stage = st.selectbox("📍 Stage", ["Sowing", "Vegetative", "Flowering", "Fruiting", "Harvesting"])
    with col3:
        location = st.selectbox("📌 Location", list(WEATHER_DATA.keys()))

    st.write("")
    left, right = st.columns([2.2, 1])

    with left:
        st.markdown("### ⚡ Smart Suggestions")
        chips = [
            "My crop leaves are yellow",
            "Pest attack in chilli",
            "Suggest fertilizer schedule",
            "Market price of cotton",
            "Rainfall coming, what to do?"
        ]

        chip_cols = st.columns(5)
        for i, chip in enumerate(chips):
            if chip_cols[i].button(chip):
                st.session_state.messages.append({"role": "user", "content": chip})
                reply = get_groq_response(chip, crop, stage, location, language, model_choice, temperature, top_k)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()

        st.write("")

        st.markdown("### 🎙️ Voice Input")
        audio = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="⏹️ Stop Recording", key="recorder")

        voice_query = None
        if audio and "bytes" in audio:
            voice_query = speech_to_text(audio["bytes"])
            if voice_query:
                st.success(f"🗣️ You said: {voice_query}")

        user_query = st.text_input("Type your query here...")
        final_query = voice_query if voice_query else user_query

        if st.button("⬆️ Send Query"):
            if final_query:
                st.session_state.messages.append({"role": "user", "content": final_query})
                reply = get_groq_response(final_query, crop, stage, location, language, model_choice, temperature, top_k)
                st.session_state.messages.append({"role": "assistant", "content": reply})

                if voice_output:
                    if language == "English":
                        voice = "en-IN-NeerjaNeural"
                    elif language == "Hindi":
                        voice = "hi-IN-SwaraNeural"
                    else:
                        voice = "te-IN-MohanNeural"

                    mp3_path = asyncio.run(generate_tts(reply, voice))
                    st.audio(mp3_path, format="audio/mp3")

                st.rerun()

        st.write("")
        st.markdown("### 🧠 Conversation")

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='ai-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

        if len(st.session_state.messages) > 0:
            pdf_file = generate_pdf_report(st.session_state.messages, crop, stage, location)
            st.download_button(
                "📥 Download Advisory PDF Report",
                data=pdf_file,
                file_name="AgriSahai_Report.pdf",
                mime="application/pdf"
            )

    with right:
        st.markdown("<div class='glass-card'><b>🌦️ Live Weather</b></div>", unsafe_allow_html=True)
        w = WEATHER_DATA[location]
        st.write(f"🌡 Temp: {w['temp']}°C")
        st.write(f"🌧 Rain: {w['rain']}%")
        st.write(f"💨 Wind: {w['wind']} km/h")
        st.write(f"🚨 Alert: {w['alert']}")

        st.write("")
        st.markdown("<div class='glass-card'><b>🧪 Spray Calculator</b></div>", unsafe_allow_html=True)

        acre = st.number_input("🌾 Land Area (acres)", 0.5, 100.0, 1.0, 0.5)
        water = st.number_input("💧 Water per acre (L)", 50.0, 400.0, 200.0, 10.0)
        dosage = st.number_input("🧴 Dosage (ml/L)", 0.0, 50.0, 2.0, 0.5)
        tank = st.number_input("🚜 Tank Capacity (L)", 5.0, 25.0, 15.0, 1.0)

        total_water = acre * water
        total_chem_ml = total_water * dosage

        st.success(f"💧 Total Water: {total_water:.1f} L")
        st.success(f"🧴 Total Chemical: {total_chem_ml:.1f} ml")
        st.info(f"🚜 Tanks Required: {(total_water / tank):.1f}")

        st.warning("⚠️ Always wear mask + gloves and avoid noon spray.")


st.markdown("---")
st.markdown("🌱 **AgriSahai AI** | Farmer Support Call Centre Assistant")
