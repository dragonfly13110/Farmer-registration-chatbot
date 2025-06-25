# app.py (ฉบับสมบูรณ์ Final - ใช้ Smart Prompting กับ Gemini)

import streamlit as st
import os
from dotenv import load_dotenv

# --- ส่วนที่ต้องใช้จาก LangChain และ Google ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

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

def get_final_answer(question: str, context: str) -> str:
    """
    ใช้ Gemini สร้างคำตอบสุดท้ายด้วย Prompt ที่ฉลาดและมีความสามารถในการให้เหตุผล
    """
    if not api_key_pool:
        return "เกิดข้อผิดพลาด: ไม่ได้ตั้งค่า Google API Key"

    # สุ่มหยิบ Key มาใช้ในแต่ละครั้ง
    import random
    selected_key = random.choice(api_key_pool)
    genai.configure(api_key=selected_key)
    
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # <<< Prompt ที่เป็นหัวใจของระบบเวอร์ชั่นนี้ >>>
    prompt = f"""
คุณคือ "ผู้เชี่ยวชาญให้คำปรึกษาด้านการขึ้นทะเบียนเกษตรกร" ที่มีความรู้จากคู่มืออย่างละเอียด และมีความสามารถในการให้เหตุผลเพื่อช่วยเหลือผู้ใช้

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

**ข้อควรจำ:** ห้ามสร้างข้อมูลที่ไม่มีอยู่ใน "ข้อมูลอ้างอิง" ขึ้นมาเองเด็ดขาด

---
**ข้อมูลอ้างอิง:**
{context}
---

**คำถามของผู้ใช้:** {question}

**คำตอบจากผู้เชี่ยวชาญ (จัดรูปแบบให้อ่านง่าย):**
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"ขออภัยครับ เกิดข้อผิดพลาดในการเชื่อมต่อกับ Gemini API: {e}"

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
    if user_question := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("กำลังค้นหาข้อมูลและสร้างคำตอบ..."):
            
            # "หว่านแห" โดยดึงเอกสารมา 7 ชิ้น (เป็นค่าที่สมดุล)
            retrieved_docs = db.similarity_search(user_question, k=7)
            
            context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # เรียกใช้ฟังก์ชันสร้างคำตอบสุดท้าย (เรียก API แค่ครั้งเดียว)
            final_answer = get_final_answer(user_question, context)
            
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            with st.chat_message("assistant"):
                st.markdown(final_answer)

            with st.expander("ดูเอกสารอ้างอิงที่ส่งให้ Gemini"):
                st.success(context)
else:
    st.warning("ระบบยังไม่พร้อมใช้งาน กรุณารอให้ Vector Store โหลดเสร็จสิ้น หรือตรวจสอบข้อผิดพลาดใน Terminal")