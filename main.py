import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType
from duckduckgo_search import DDGS, DuckDuckGoSearchException
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import requests
import os
import re
import time

# .env dosyasını yükle
load_dotenv()

# Sayfa başlığı
st.set_page_config(page_title="LegolasAI", layout="wide")
st.title("🧝‍♂️ LegolasAI")

# Session state başlat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""

# Sidebar
with st.sidebar:
    st.subheader("📂 PDF Yükle")
    file = st.file_uploader("Bir PDF yükle (İsteğe bağlı)", type="pdf")

    st.markdown("---")
    st.subheader("🌐 Türkçe Çeviri")
    enable_translation = st.checkbox("🔁 Otomatik TR ⇄ EN", value=True)

    st.markdown("---")
    st.subheader("🕑 Chat Geçmişi")
    if st.session_state.chat_history:
        for role, msg in st.session_state.chat_history[::-1][:10]:
            st.markdown(f"**{role.title()}:** {msg}")
    if st.button("🧹 Geçmişi Temizle"):
        st.session_state.chat_history = []

# PDF işle
if file:
    reader = PdfReader(file)
    raw_text = "".join(page.extract_text() or "" for page in reader.pages)

    if raw_text != st.session_state.doc_text:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_text(raw_text)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        st.session_state.vectorstore = vectorstore
        st.session_state.doc_text = raw_text
        st.success("✅ PDF yüklendi.")

retriever = (
    st.session_state.vectorstore.as_retriever()
    if st.session_state.vectorstore
    else None
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
)

# Web Search Tool (rate limit korumalı)
def web_search(query, retries=3, delay=2):
    for attempt in range(retries):
        try:
            results = DDGS().text(query)
            if not results:
                return "🔍 Arama sonucu bulunamadı."
            return "\n".join([f"🔗 {r['title']} - {r['href']}" for r in results[:3]])
        except DuckDuckGoSearchException as e:
            if "Ratelimit" in str(e):
                time.sleep(delay)
            else:
                return f"⚠️ Arama hatası: {str(e)}"
        except Exception as e:
            return f"⚠️ Web araması sırasında bir hata oluştu: {str(e)}"
    return "⚠️ Arama limiti aşıldı. Lütfen daha sonra tekrar deneyin."

# Calculator Tool
def calculator(query):
    try:
        return str(eval(query))
    except:  # noqa: E722
        return "⚠️ Geçersiz işlem."

# Rüzgar yönü
def derece_to_yon(degree):
    directions = [
        "Kuzey",
        "Kuzeydoğu",
        "Doğu",
        "Güneydoğu",
        "Güney",
        "Güneybatı",
        "Batı",
        "Kuzeybatı",
    ]
    ix = int((degree + 22.5) // 45) % 8
    return directions[ix]

# Hava Durumu Tool
def hava_durumu_tool(query: str) -> str:
    api_key = os.getenv("WEATHER_API_KEY")
    şehir = query.lower().replace("hava", "").replace("durumu", "")
    şehir = re.sub(r"[^a-zA-ZğüşöçıİĞÜŞÖÇ\s]", "", şehir).strip()
    if not şehir:
        return "⚠️ Lütfen bir şehir ismi belirtin."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={şehir}&appid={api_key}&lang=tr&units=metric"
    try:
        res = requests.get(url).json()
        if res.get("cod") != 200:
            return f"❌ Hava durumu alınamadı: {res.get('message', '')}"

        desc = res["weather"][0]["description"].capitalize()
        temp = res["main"]["temp"]
        feels = res["main"]["feels_like"]
        humidity = res["main"]["humidity"]
        wind_speed = res["wind"]["speed"]
        wind_deg = res["wind"].get("deg", 0)
        wind_dir = derece_to_yon(wind_deg)

        return (
            f"📍 {şehir.title()} için hava durumu:\n"
            f"🌡️ Sıcaklık: {temp}°C (Hissedilen: {feels}°C)\n"
            f"🌥️ Durum: {desc}\n"
            f"💧 Nem: {humidity}%\n"
            f"💨 Rüzgar: {wind_speed} m/s, {wind_dir} yönünden"
        )
    except Exception as e:
        return f"⚠️ API hatası: {str(e)}"

# Araçlar listesi
tools = [
    Tool(name="WebSearch", func=web_search, description="Web'den bilgi arar."),
    Tool(name="Calculator", func=calculator, description="Hesaplama yapar."),
    Tool(
        name="HavaDurumu",
        func=hava_durumu_tool,
        description="Gerçek zamanlı hava durumu verir.",
    ),
]

# Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=False,
)

# RAG (belge varsa)
if retriever:
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory, return_source_documents=False
    )

# Kullanıcı sorusu
user_query = st.chat_input("💬 Sorunuzu yazın...")

if user_query:
    with st.spinner("🤖 Yanıt hazırlanıyor..."):
        translated_query = (
            GoogleTranslator(source="auto", target="en").translate(user_query)
            if enable_translation
            else user_query
        )

        tool_keywords = [
            "kaç",
            "hesapla",
            "arama",
            "ara",
            "google",
            "internet",
            "hava",
            "sıcaklık",
            "bugün hava",
            "derece",
            "nasıl hava",
        ]

        if any(k in user_query.lower() for k in tool_keywords):
            response_en = agent.run(translated_query)
        elif retriever:
            response_en = rag_chain.run({"question": translated_query})
        else:
            response_en = llm.invoke(translated_query).content

        response = (
            GoogleTranslator(source="en", target="tr").translate(response_en)
            if enable_translation
            else response_en
        )

        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("assistant", response))

# Sohbet geçmişini göster
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
