# app.py (ฉบับสมบูรณ์ Final - ใช้ Smart Prompting กับ Gemini)

import streamlit as st
import os
import random
from dotenv import load_dotenv

# --- ส่วนที่ต้องใช้จาก LangChain และ Google (ปรับปรุงใหม่) ---
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# LangChain components for conversational memory and RAG
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage # For converting Streamlit history to LangChain messages
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser # To parse the output from the LLM

# --- โหลดค่าตั้งค่าและโมเดล ---
load_dotenv()

# ตั้งค่า Google API Key (ใช้ระบบ Key Rotation ที่เราทำไว้)
api_keys = [
    os.getenv("GOOGLE_API_KEY_1"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3"),
]
api_key_pool = [key for key in api_keys if key]

if not api_key_pool:
    st.error("ไม่พบ Google API Key ใดๆ ในไฟล์ .env! กรุณาตรวจสอบ")

VECTORSTORE_PATH = "vectorstore_combined"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# --- ฟังก์ชันหลัก ---

@st.cache_resource
def load_vector_store():
    """โหลด Vector Store ที่สร้างไว้แล้ว (ทำครั้งเดียว)"""
    if not os.path.exists(VECTORSTORE_PATH):
        st.error(f"ไม่พบฐานข้อมูล Vector Store ที่ '{VECTORSTORE_PATH}'! กรุณารันไฟล์ `1_prepare_vectorstore.py` ก่อน")
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db = FAISS.load_local(
            VECTORSTORE_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลด Vector Store: {e}")
        return None

# Store chat history outside the chain for Streamlit's session state
# This is a simple in-memory store for a single user session.
# For multi-user applications, this would need to be more sophisticated (e.g., database).
store = {}
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def create_chains():
    """
    สร้าง LangChain runnables สองชุด:
    1. rewriter_chain: สำหรับแปลงคำถามของผู้ใช้ให้เป็น search query ที่ดีขึ้น
    2. answer_chain: สำหรับสร้างคำตอบสุดท้ายจาก context และคำถาม
    """
    if not api_key_pool:
        st.error("ไม่พบ Google API Key ใดๆ ในไฟล์ .env! ไม่สามารถสร้างโมเดลได้")
        return None, None

    # สร้าง LLM instance เดียวเพื่อใช้ร่วมกัน
    selected_key = random.choice(api_key_pool)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.7,
        google_api_key=selected_key,
    )

    # 1. Chain สำหรับการแปลงคำถาม (Query Transformation)
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """คุณคือผู้ช่วย AI ที่เชี่ยวชาญในการแปลงคำถามของผู้ใช้ให้เป็นคำค้นหา (Search Query) ที่มีประสิทธิภาพสำหรับ Vector Database
ภารกิจของคุณคือการอ่านบทสนทนาล่าสุดและคำถามติดตามผล แล้วสร้างคำค้นหาที่เป็นประโยคสมบูรณ์ (Standalone Question) ที่เหมาะสำหรับการค้นหาข้อมูลในคู่มือการขึ้นทะเบียนเกษตรกร

- คำค้นหาควรจะชัดเจนและตรงประเด็น
- แปลงภาษาพูดให้เป็นภาษาที่เป็นทางการมากขึ้น (เช่น "ทำสวน" -> "การเพาะปลูกพืช", "ต้องใช้อะไรบ้าง" -> "เอกสารและคุณสมบัติที่จำเป็น")
- รวมบริบทที่สำคัญจากบทสนทนาก่อนหน้าเข้ามาในคำค้นหาใหม่

ตัวอย่าง:
ประวัติการสนทนา:
Human: ผมปลูกทุเรียน 10 ไร่ครับ
AI: รับทราบครับ การปลูกทุเรียนจัดเป็นไม้ผล มีข้อมูลอะไรให้ช่วยไหมครับ
คำถามติดตามผล:
แล้วต้องใช้เอกสารอะไรบ้างครับ

คำค้นหาที่ควรสร้าง:
"เอกสารที่จำเป็นสำหรับการขึ้นทะเบียนเกษตรกรผู้ปลูกทุเรียนมีอะไรบ้าง"
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    
    rewriter_chain_base = rewrite_prompt | llm | StrOutputParser()
    rewriter_chain = RunnableWithMessageHistory(
        rewriter_chain_base,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    # 2. Chain สำหรับการสร้างคำตอบสุดท้าย (RAG)
    # Prompt นี้จะรับ context ที่ค้นหามาแล้ว, คำถามดั้งเดิม, และประวัติการสนทนา
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """คุณคือ "ผู้เชี่ยวชาญให้คำปรึกษาด้านการขึ้นทะเบียนเกษตรกร" ที่มีความรู้จากคู่มืออย่างละเอียด และมีความสามารถในการให้เหตุผลเพื่อช่วยเหลือผู้ใช้

**ภารกิจของคุณ:** ตอบคำถามของผู้ใช้ให้ถูกต้อง, เป็นประโยชน์, กระชับ, และอ่านง่ายที่สุด โดยมีขั้นตอนการคิดและตอบดังนี้:

**ขั้นตอนที่ 1: ตีความคำถาม (Intent Recognition)**
- อ่าน "คำถามของผู้ใช้" และทำความเข้าใจเจตนาที่แท้จริง
- ตีความคำศัพท์ภาษาพูดให้เป็นคำที่เป็นทางการ (เช่น "สวนเงาะ" หมายถึง "ไม้ผล")

**ขั้นตอนที่ 2: ค้นหาและตรวจสอบกับข้อมูลอ้างอิง (Fact-Checking)**
- นำความเข้าใจจากขั้นตอนที่ 1 ไปค้นหาข้อมูลที่เกี่ยวข้องที่สุดใน "ข้อมูลอ้างอิง" ที่ให้มา
- ยึดข้อมูลใน "ข้อมูลอ้างอิง" เป็นความจริงสูงสุดเสมอ

**ขั้นตอนที่ 3: สร้างคำตอบพร้อมจัดรูปแบบ (Formatted Response Generation)**
- สร้างคำตอบตามกรณีด้านล่างนี้ และ **ต้องจัดรูปแบบคำตอบให้อ่านง่ายเสมอ**
- **กฎการจัดรูปแบบ:**
  - **กระชับ:** ใช้ประโยคสั้นๆ ตรงประเด็น
  - **ย่อหน้า:** หากคำตอบมีหลายประเด็น ให้แบ่งเป็นย่อหน้าสั้นๆ
  - **หัวข้อ/Bullet Points:** หากเป็นการให้ข้อมูลหลายข้อ ให้ใช้หัวข้อสั้นๆ หรือ bullet points (เครื่องหมาย -) เพื่อแยกประเด็น
  - **อีโมจิ:** ใช้อีโมจิที่เกี่ยวข้อง 1-2 ตัวต่อหนึ่งคำตอบ เพื่อให้ดูเป็นมิตร (เช่น ✅, ❌, 📋, 🌾, 💡) แต่อย่าใช้เยอะเกินไป

 - **แนวทางการตอบ:**
  - **กรณีที่ 1 (พบข้อมูลตรงๆ):** สรุปข้อมูลที่พบมาตอบโดยตรง พร้อมจัดรูปแบบให้อ่านง่าย
  - **กรณีที่ 2 (พบข้อมูลใกล้เคียง):** นำข้อมูลที่พบมาตอบ แล้วอาจจะเสริมว่า "ซึ่งเกณฑ์นี้ใช้กับไม้ผลโดยทั่วไป รวมถึงเงาะด้วยครับ"
  - **กรณีที่ 3 (ไม่พบข้อมูลโดยตรง แต่มีข้อมูลเกี่ยวข้อง):** เริ่มต้นว่า "ต้องขออภัยครับ 😥 จากการตรวจสอบ ไม่พบข้อมูลสำหรับกรณีของคุณโดยเฉพาะ แต่มีหลักเกณฑ์ทั่วไปที่เกี่ยวข้องดังนี้ครับ:" จากนั้นให้ข้อมูลนั้นในรูปแบบ bullet points
  - **กรณีที่ 4 (ไม่พบข้อมูลเลย):** ตอบอย่างสุภาพว่า: "ต้องขออภัยครับ ไม่พบข้อมูลที่เกี่ยวข้องกับเรื่องนี้ในคู่มือเลย 📂 เพื่อความถูกต้องที่สุด ผมแนะนำให้ลองสอบถามโดยตรงกับเจ้าหน้าที่ ณ สำนักงานเกษตรอำเภอใกล้บ้านท่านนะครับ"
  - **กรณีที่ 5 (คำถามไม่เกี่ยวข้อง):** ตอบว่า "ต้องขออภัยครับ คำถามนี้ไม่เกี่ยวข้องกับการขึ้นทะเบียนเกษตรกร"
**ข้อควรจำ:** ห้ามสร้างข้อมูลที่ไม่มีอยู่ใน "ข้อมูลอ้างอิง" ขึ้นมาเองเด็ดขาด
**สำคัญมาก:** หากข้อมูลใน "ข้อมูลอ้างอิง" ไม่เพียงพอต่อการตอบคำถามโดยตรง ให้ตอบตาม "แนวทางการตอบ" กรณีที่ 3 หรือ 4 อย่างเคร่งครัด ห้ามใช้ความรู้ทั่วไปมาตอบแทนข้อมูลอ้างอิง

---
**ข้อมูลอ้างอิง:**
{context}
---
"""),
            MessagesPlaceholder(variable_name="chat_history"), # This is where previous messages go
            ("human", "{question}")
        ]
    )

    answer_chain_base = answer_prompt | llm | StrOutputParser()
    answer_chain = RunnableWithMessageHistory(
        answer_chain_base,
        get_session_history,
        input_messages_key="question", # This will also receive 'context'
        history_messages_key="chat_history",
    )

    return rewriter_chain, answer_chain

# --- ส่วนของหน้าเว็บ (Streamlit UI) ---
st.set_page_config(page_title="เกษตรกรแชตบอท", page_icon="👩‍🌾")
st.title("👩‍🌾 แชตบอทถาม-ตอบเรื่องการขึ้นทะเบียนเกษตรกร")
st.write("ขับเคลื่อนโดย Google Gemini และทะเบียนเกษตรกร ปี 2568 กรมส่งเสริมการเกษตร")

db = load_vector_store()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Logic ---
if db:
    # สร้าง chains ทั้งสองและเก็บไว้ใน session state เพื่อไม่ให้สร้างใหม่ทุกครั้ง
    if 'rewriter_chain' not in st.session_state or 'answer_chain' not in st.session_state:
        st.session_state.rewriter_chain, st.session_state.answer_chain = create_chains()

    if user_question := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # ขั้นตอนที่ 1: ใช้ Gemini ช่วยแปลงคำถามให้ดีขึ้น (Query Transformation)
        with st.spinner("กำลังวิเคราะห์และเรียบเรียงคำถามของคุณ..."):
            try:
                rewritten_question = st.session_state.rewriter_chain.invoke(
                    {"question": user_question},
                    config={"configurable": {"session_id": "streamlit_chat_session"}}
                )
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์คำถาม: {e}")
                st.stop()

        # ขั้นตอนที่ 2: ค้นหาข้อมูลด้วยคำถามที่แปลงแล้ว และสร้างคำตอบ
        with st.spinner("กำลังค้นหาข้อมูลและสร้างคำตอบ..."):
            # "หว่านแห" โดยดึงเอกสารมา 10 ชิ้น
            retrieved_docs = db.similarity_search(rewritten_question, k=10)

            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

            # ขั้นตอนที่ 3: เรียก Chain สร้างคำตอบสุดท้าย
            try:
                final_answer = st.session_state.answer_chain.invoke(
                    {"question": user_question, "context": context}, # ส่งคำถามดั้งเดิมและ context
                    config={"configurable": {"session_id": "streamlit_chat_session"}}, # Use a fixed session ID for this single-user app
                )
            except Exception as e:
                final_answer = f"ขออภัยครับ เกิดข้อผิดพลาดในการเชื่อมต่อกับ Gemini API หรือการประมวลผล: {e}"
                st.error(final_answer) # Display error in Streamlit

            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            with st.chat_message("assistant"):
                st.markdown(final_answer)

            with st.expander("ดูขั้นตอนการทำงานของ AI"):
                st.info(f"**คำค้นหาที่ AI สร้างขึ้นเพื่อค้นหาข้อมูล:**\n\n> {rewritten_question}")
                st.success(f"**ข้อมูลอ้างอิงที่พบและส่งให้ AI พิจารณา:**\n\n---\n{context}")
else:
    st.warning("ระบบยังไม่พร้อมใช้งาน กรุณารอให้ Vector Store โหลดเสร็จสิ้น หรือตรวจสอบข้อผิดพลาดใน Terminal")