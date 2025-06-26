import streamlit as st
import os
import random
from dotenv import load_dotenv

# --- ส่วนที่ต้องใช้จาก LangChain และ Google ---
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# --- โหลดค่าตั้งค่าและโมเดล ---
load_dotenv()
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

# --- ฟังก์ชันหลัก (Cached) ---

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

store = {}
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """ดึงประวัติการแชตสำหรับ session ปัจจุบัน"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

@st.cache_resource
def get_chains():
    """สร้าง LangChain chains ทั้งหมดและเก็บไว้ใน cache เพื่อไม่ให้สร้างใหม่ทุกครั้ง"""
    st.write("กำลังเตรียมผู้ช่วย AI...") # แสดงให้ผู้ใช้เห็นตอนโหลดครั้งแรก
    if not api_key_pool:
        return None, None
        
    selected_key = random.choice(api_key_pool)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.7,
        google_api_key=selected_key,
    )

    # Chain 1: สำหรับสร้างคำตอบสุดท้าย (RAG)
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """คุณคือ "ผู้เชี่ยวชาญให้คำปรึกษาด้านการขึ้นทะเบียนเกษตรกร" ที่มีความรู้จากคู่มืออย่างละเอียด และมีความสามารถในการให้เหตุผลเพื่อช่วยเหลือผู้ใช้
คุณคือ "ผู้ช่วย AI เกษตรกรอัจฉริยะ" ที่มีความรู้ลึกซึ้งเกี่ยวกับการขึ้นทะเบียนเกษตรกรจากคู่มือทางการ และยังมีความสามารถในการสนทนาทั่วไปอย่างเป็นธรรมชาติและเป็นมิตร

**ภารกิจของคุณ:** ตอบคำถามของผู้ใช้ให้ถูกต้อง, เป็นประโยชน์, กระชับ, และอ่านง่ายที่สุด โดยมีขั้นตอนการคิดและตอบดังนี้:

**ขั้นตอนที่ 1: ตีความเจตนาและบริบท (Intent & Context Understanding)**
- อ่าน "คำถามของผู้ใช้" และ "ประวัติการสนทนา" เพื่อทำความเข้าใจเจตนาที่แท้จริงและบริบทของการสนทนา
- หากเป็นคำถามเกี่ยวกับการขึ้นทะเบียนเกษตรกร ให้ตีความคำศัพท์ภาษาพูดให้เป็นคำที่เป็นทางการ (เช่น "สวนเงาะ" หมายถึง "ไม้ผล")
- หากเป็นการทักทายหรือการสนทนาทั่วไป ให้ตอบกลับอย่างเป็นมิตรและเป็นธรรมชาติ

**ขั้นตอนที่ 2: ค้นหาและตรวจสอบข้อมูล (Information Retrieval & Validation)**
- **สำหรับคำถามเกี่ยวกับการขึ้นทะเบียนเกษตรกร:**
    - นำความเข้าใจจากขั้นตอนที่ 1 ไปค้นหาข้อมูลที่เกี่ยวข้องที่สุดใน "ข้อมูลอ้างอิง" ที่ให้มา
    - ยึดข้อมูลใน "ข้อมูลอ้างอิง" เป็นความจริงสูงสุดเสมอ
    - หาก "ข้อมูลอ้างอิง" ไม่เพียงพอหรือไม่เกี่ยวข้องโดยตรง ให้ใช้ความรู้ทั่วไปของคุณ (General Knowledge) เพื่อให้คำตอบที่เป็นประโยชน์ แต่ต้องระบุให้ชัดเจนว่าข้อมูลนั้นไม่ได้มาจากคู่มือ
- **สำหรับคำทักทายหรือการสนทนาทั่วไป:**
    - ใช้ความรู้ทั่วไปของคุณในการตอบกลับอย่างเหมาะสม ไม่จำเป็นต้องอ้างอิงจากคู่มือ

**ขั้นตอนที่ 3: สร้างคำตอบพร้อมจัดรูปแบบ (Formatted Response Generation)**
- สร้างคำตอบตามกรณีด้านล่างนี้ และ **ต้องจัดรูปแบบคำตอบให้อ่านง่ายเสมอ**
- **กฎการจัดรูปแบบ:**
  - **กระชับ:** ใช้ประโยคสั้นๆ ตรงประเด็น
  - **ย่อหน้า:** หากคำตอบมีหลายประเด็น ให้แบ่งเป็นย่อหน้าสั้นๆ
  - **หัวข้อ/Bullet Points:** หากเป็นการให้ข้อมูลหลายข้อ ให้ใช้หัวข้อสั้นๆ หรือ bullet points (เครื่องหมาย -) เพื่อแยกประเด็น
  - **อีโมจิ:** ใช้อีโมจิที่เกี่ยวข้อง 1-2 ตัวต่อหนึ่งคำตอบ เพื่อให้ดูเป็นมิตร (เช่น ✅, ❌, 📋, 🌾, 💡, 👋, 😊) แต่อย่าใช้เยอะเกินไป

 - **แนวทางการตอบ:**
  - **กรณีที่ 1 (คำถามเกี่ยวกับการขึ้นทะเบียนเกษตรกร และพบข้อมูลในข้อมูลอ้างอิง):** สรุปข้อมูลที่พบมาตอบโดยตรง พร้อมจัดรูปแบบให้อ่านง่าย
  - **กรณีที่ 2 (คำถามเกี่ยวกับการขึ้นทะเบียนเกษตรกร แต่ข้อมูลอ้างอิงไม่เพียงพอ/ไม่เกี่ยวข้องโดยตรง):**
    - เริ่มต้นว่า "💡 จากความรู้ทั่วไปของผม/ดิฉัน..." หรือ "🤔 แม้ว่าในคู่มือจะไม่มีข้อมูลโดยตรง แต่โดยทั่วไปแล้ว..."
    - ให้ข้อมูลที่เป็นประโยชน์จากความรู้ทั่วไปของคุณ
    - อาจเสริมว่า "หากต้องการข้อมูลที่แม่นยำที่สุดสำหรับกรณีของคุณ ผมแนะนำให้สอบถามโดยตรงกับเจ้าหน้าที่ ณ สำนักงานเกษตรอำเภอใกล้บ้านท่านนะครับ"
  - **กรณีที่ 3 (คำถามไม่เกี่ยวข้องกับการขึ้นทะเบียนเกษตรกร เช่น คำทักทาย, สภาพอากาศ, หรือเรื่องทั่วไป):**
    - ตอบกลับอย่างเป็นมิตรและเป็นธรรมชาติ
    - อาจจะถามกลับเบาๆ เพื่อชวนคุย หรือเปลี่ยนกลับมาที่หัวข้อการขึ้นทะเบียนเกษตรกร
    - ตัวอย่าง: "สวัสดีครับ! 😊 มีอะไรให้ผมช่วยเรื่องการขึ้นทะเบียนเกษตรกรไหมครับ?" หรือ "อากาศดีนะครับวันนี้! ☀️ มีคำถามเกี่ยวกับการเกษตรหรือการขึ้นทะเบียนไหมครับ?"
  - **ข้อควรจำ:** ห้ามสร้างข้อมูลที่ไม่มีอยู่ใน "ข้อมูลอ้างอิง" *และ* ไม่ใช่ความรู้ทั่วไปที่สมเหตุสมผลขึ้นมาเองเด็ดขาด

----
**ข้อมูลอ้างอิง:**
{context}
----
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ]
    )
    answer_chain_base = answer_prompt | llm | StrOutputParser()
    answer_chain = RunnableWithMessageHistory(answer_chain_base, get_session_history, input_messages_key="question", history_messages_key="chat_history")

    # Chain 2: สำหรับสร้างคำถามแนะนำ
    suggestion_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """คุณคือผู้ช่วย AI ที่มีประโยชน์ หน้าที่ของคุณคือการดูบทสนทนาล่าสุด และสร้างคำถามติดตามผลที่เกี่ยวข้อง 3 ข้อที่ผู้ใช้อาจจะถามต่อไป
- คำถามควรสั้นและกระชับ
- คำถามควรเกี่ยวข้องกับการขึ้นทะเบียนเกษตรกร หรือหัวข้อที่กำลังสนทนากันอยู่
- ตอบกลับเฉพาะคำถาม 3 ข้อเท่านั้น โดยแต่ละข้อขึ้นบรรทัดใหม่
- ห้ามใส่ข้อความอื่นใดๆ นอกเหนือจากคำถาม เช่น ห้ามใส่ "นี่คือคำถามแนะนำ:" หรือใช้เครื่องหมาย bullet point

ตัวอย่าง:
ประวัติการสนทนา:
Human: ผมปลูกทุเรียน 10 ไร่ครับ
AI: รับทราบครับ การปลูกทุเรียนจัดเป็นไม้ผล มีข้อมูลอะไรให้ช่วยไหมครับ

คำถามแนะนำที่ควรสร้าง:
ต้องใช้เอกสารอะไรบ้างในการขึ้นทะเบียน?
ขึ้นทะเบียนได้ที่ไหน?
มีค่าใช้จ่ายในการขึ้นทะเบียนหรือไม่?"""),
            MessagesPlaceholder(variable_name="chat_history"),
        ]
    )
    suggestion_chain = suggestion_prompt | llm | StrOutputParser()
    
    return answer_chain, suggestion_chain

# --- UI และ Logic หลัก (โครงสร้างใหม่ทั้งหมด) ---
st.set_page_config(page_title="เกษตรกรแชตบอท", page_icon="👩‍🌾")
st.title("👩‍🌾 แชตบอทถาม-ตอบเรื่องการขึ้นทะเบียนเกษตรกร")
st.write("ขับเคลื่อนโดย Google Gemini และทะเบียนเกษตรกร ปี 2568 กรมส่งเสริมการเกษตร")

db = load_vector_store()

if db:
    # โหลด Chains จาก Cache (จะทำงานแค่ครั้งแรก)
    answer_chain, suggestion_chain = get_chains()
    if not answer_chain:
        st.error("ไม่สามารถสร้าง AI chains ได้ กรุณาตรวจสอบ API Key")
        st.stop()

    # --- ส่วนที่ 1: การจัดการ Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = []

    # --- ส่วนที่ 2: การแสดงผล UI ที่มีอยู่แล้ว ---
    # แสดงประวัติการแชตทั้งหมด
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # แสดงปุ่มคำถามแนะนำ (ถ้ามี)
    if st.session_state.suggested_questions:
        cols = st.columns(len(st.session_state.suggested_questions))
        for i, suggestion in enumerate(st.session_state.suggested_questions):
            if cols[i].button(suggestion, key=f"suggestion_{i}"):
                # เมื่อกดปุ่ม ให้เก็บคำถามไว้ใน state เพื่อนำไปประมวลผล
                st.session_state.user_input = suggestion
                st.rerun() # สั่งให้แอปทำงานใหม่เพื่อประมวลผล input นี้

    # --- ส่วนที่ 3: การรับ Input ใหม่ ---
    # รับ input จากช่องแชตหลัก
    if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):
        st.session_state.user_input = prompt
        st.rerun() # สั่งให้แอปทำงานใหม่เพื่อประมวลผล input นี้

    # --- ส่วนที่ 4: การประมวลผลหลัก (ทำงานเมื่อมี Input ใหม่เท่านั้น) ---
    if "user_input" in st.session_state and st.session_state.user_input:
        
        # ดึง input มาใช้แล้วลบทิ้งทันที เพื่อป้องกันการรันซ้ำ
        user_input = st.session_state.user_input
        del st.session_state.user_input

        # ล้างคำแนะนำเก่าทิ้งทันที
        st.session_state.suggested_questions = []

        # แสดงข้อความของผู้ใช้และเพิ่มเข้า history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # แสดงสถานะกำลังทำงานของ AI
        with st.chat_message("assistant"):
            with st.spinner("สักครู่นะครับ กำลังค้นหาและเรียบเรียงคำตอบ..."):
                # ขั้นตอนที่ 1: ค้นหาข้อมูล
                retrieved_docs = db.similarity_search(user_input, k=5)
                context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

                # ขั้นตอนที่ 2: สร้างคำตอบ
                final_answer = ""
                try:
                    final_answer = answer_chain.invoke(
                        {"question": user_input, "context": context},
                        config={"configurable": {"session_id": "streamlit_chat_session"}},
                    )
                except Exception as e:
                    final_answer = f"ขออภัยครับ เกิดข้อผิดพลาดในการเชื่อมต่อกับ Gemini: {e}"
                    st.error(final_answer)

                # แสดงคำตอบของ AI
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

                # ขั้นตอนที่ 3: สร้างคำแนะนำใหม่
                try:
                    chat_history_for_suggestion = get_session_history("streamlit_chat_session").messages
                    if chat_history_for_suggestion:
                        suggestion_text = suggestion_chain.invoke({"chat_history": chat_history_for_suggestion})
                        suggestions = [s.strip() for s in suggestion_text.split('\n') if s.strip()]
                        suggestions = [s.lstrip('-* ').strip() for s in suggestions if len(s) > 10]
                        # [จุดสำคัญ] บันทึกคำแนะนำใหม่ลงใน session state
                        st.session_state.suggested_questions = suggestions[:3]
                except Exception as e:
                    print(f"ไม่สามารถสร้างคำถามแนะนำได้: {e}")
                    st.session_state.suggested_questions = [] # ถ้าพลาดก็ให้เป็นค่าว่าง
        
        # [จุดสำคัญ] สั่ง rerun ครั้งสุดท้าย เพื่อวาด UI ใหม่ทั้งหมด
        # ซึ่งในรอบใหม่นี้ ปุ่มคำถามแนะนำจะปรากฏขึ้น!
        st.rerun()

else:
    st.warning("ระบบยังไม่พร้อมใช้งาน กรุณารอให้ Vector Store โหลดเสร็จสิ้น หรือตรวจสอบข้อผิดพลาดใน Terminal")