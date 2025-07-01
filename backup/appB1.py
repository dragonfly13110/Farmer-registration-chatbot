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

VECTORSTORE_PATH = "vectorstore_smart_chunking_v2"
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
    """สร้าง Chains ทั้งหมด (รวม Rewriter)"""
    st.write("กำลังเตรียมผู้ช่วย AI...")
    if not api_key_pool:
        return None, None
        
    selected_key = random.choice(api_key_pool)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite-preview-06-17",
        temperature=0.7,
        google_api_key=selected_key,
    )

    # ⬇️ [การเปลี่ยนแปลง] Chain 1: สำหรับแปลงคำถาม (Rewriter)
    rewriter_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """คุณคือผู้ช่วย AI ที่เชี่ยวชาญในการแปลงคำถามของผู้ใช้ให้เป็นคำค้นหาที่มีประสิทธิภาพสำหรับ Vector Database
ภารกิจของคุณคือการอ่านบทสนทนาล่าสุดและคำถามติดตามผล แล้วสร้างคำค้นหาที่เป็นประโยคสมบูรณ์ ที่เหมาะสำหรับการค้นหาข้อมูลในคู่มือการขึ้นทะเบียนเกษตรกร

- คำค้นหาควรจะชัดเจนและตรงประเด็น
- แปลงภาษาพูดให้เป็นภาษาที่เป็นทางการมากขึ้น (เช่น "ทำสวน" -> "การเพาะปลูกพืช", "ต้องใช้อะไรบ้าง" -> "เอกสารและคุณสมบัติที่จำเป็น")
- รวมบริบทที่สำคัญจากบทสนทนาก่อนหน้าเข้ามาในคำค้นหาใหม่
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    rewriter_chain = rewriter_prompt | llm | StrOutputParser()


    # ⬇️ [การเปลี่ยนแปลง] Chain 2: สำหรับสร้างคำตอบ (ใช้ Prompt ที่รวมการสร้างคำแนะนำแล้ว)
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """คุณคือ "ผู้เชี่ยวชาญให้คำปรึกษาด้านการขึ้นทะเบียนเกษตรกร" ที่มีความเป็นมิตร

**ภารกิจของคุณ:** ตอบคำถามของผู้ใช้ให้ถูกต้องและอ่านง่ายที่สุด โดยทำตามขั้นตอนต่อไปนี้อย่างเคร่งครัด:

**ขั้นตอนที่ 1-3: การสร้างคำตอบหลัก**
- อ่าน "คำถามของผู้ใช้" และ "ประวัติการสนทนา" เพื่อทำความเข้าใจเจตนา
- ค้นหาข้อมูลที่เกี่ยวข้องที่สุดใน "ข้อมูลอ้างอิง" ที่ให้มา และยึดข้อมูลนั้นเป็นหลัก
- หากข้อมูลอ้างอิงไม่พอ ให้ใช้ความรู้ทั่วไป แต่ต้องแจ้งให้ผู้ใช้ทราบ
- สร้างคำตอบหลักให้เสร็จสมบูรณ์ โดยใช้ภาษาที่เป็นมิตร จัดรูปแบบด้วย Bullet points หรือย่อหน้าสั้นๆ เพื่อให้อ่านง่าย

**ขั้นตอนที่ 4: สร้างแนวทางคำถามต่อไป (สำคัญมาก)**
- **หลังจาก**ตอบคำถามหลักเสร็จสิ้นแล้ว ให้คุณเว้นบรรทัด 2 บรรทัด
- จากนั้นสร้างส่วน "แนวทางคำถาม" โดยเริ่มต้นด้วยข้อความว่า "**💡 ลองถามต่อได้เลย:**"
- ตามด้วยรายการคำถาม 3 ข้อ โดยแต่ละข้อขึ้นบรรทัดใหม่และใช้เครื่องหมายขีด (-) นำหน้า
- **กฎการสร้างคำถามแนะนำ:**
  - **ต้องเกี่ยวข้องกับ "การขึ้นทะเบียนเกษตรกร" โดยตรงเท่านั้น** (เช่น เอกสาร, สถานที่, คุณสมบัติ, ขั้นตอน, สิทธิประโยชน์)
  - **ต้องเป็นคำถามที่ต่อยอดจากการสนทนา** และยังไม่มีการถาม-ตอบกันในประวัติการแชต
  - แม้ว่าผู้ใช้จะทักทายหรือชวนคุยเรื่องทั่วไป คำถามแนะนำก็ยังต้องเป็นเรื่องการขึ้นทะเบียนเกษตรกร เพื่อดึงการสนทนากลับเข้าสู่หัวข้อหลัก

**ตัวอย่างผลลัพธ์ที่คาดหวัง:**
[ส่วนคำตอบหลักที่ AI สร้างขึ้นตามปกติ]... การขึ้นทะเบียนประเภทไม้ผลไม่มีค่าใช้จ่ายครับ

**💡 ลองถามต่อได้เลย:**
- ต้องใช้เอกสารอะไรบ้าง?
- ขึ้นทะเบียนได้ที่ไหน?
- ถ้าปลูกพืชหลายชนิด ต้องทำอย่างไร?

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
    answer_chain = RunnableWithMessageHistory(
        answer_chain_base, 
        get_session_history, 
        input_messages_key="question", 
        history_messages_key="chat_history"
    )
    
    return rewriter_chain, answer_chain

# --- UI และ Logic หลัก ---
st.set_page_config(page_title="เกษตรกรแชตบอท", page_icon="👩‍🌾")
st.title("👩‍🌾 แชตบอทถาม-ตอบเรื่องการขึ้นทะเบียนเกษตรกร")
st.write("ขับเคลื่อนโดย Google Gemini และทะเบียนเกษตรกร ปี 2568 กรมส่งเสริมการเกษตร")

db = load_vector_store()

if db:
    # ⬇️ [การเปลี่ยนแปลง] โหลด 2 Chains
    rewriter_chain, answer_chain = get_chains()
    if not answer_chain:
        st.error("ไม่สามารถสร้าง AI chain ได้ กรุณาตรวจสอบ API Key")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # ⬇️ [การเปลี่ยนแปลง] แบ่งการทำงานเป็น 3 ขั้นตอนที่ชัดเจน
            rewritten_question = ""
            context = ""
            final_answer_with_suggestions = ""

            with st.spinner("กำลังทำความเข้าใจคำถามของคุณ..."):
                # ขั้นตอนที่ 1: แปลงคำถาม
                try:
                    chat_history = get_session_history("streamlit_chat_session").messages
                    rewritten_question = rewriter_chain.invoke({
                        "chat_history": chat_history,
                        "question": user_input
                    })
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการแปลงคำถาม: {e}")
                    st.stop()

            with st.spinner("กำลังค้นหาข้อมูลที่เกี่ยวข้อง..."):
                # ขั้นตอนที่ 2: ค้นหาด้วยคำถามที่แปลงแล้ว
                retrieved_docs = db.similarity_search(rewritten_question, k=7)
                context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            
            with st.spinner("กำลังเรียบเรียงคำตอบ..."):
                # ขั้นตอนที่ 3: สร้างคำตอบ
                try:
                    # **สำคัญ:** เราส่ง user_input (คำถามดั้งเดิม) ให้ AI ตอบ เพื่อให้มันรู้ว่าผู้ใช้ถามอะไรจริงๆ
                    # แต่เราใช้ context ที่ได้จาก rewritten_question
                    final_answer_with_suggestions = answer_chain.invoke(
                        {"question": user_input, "context": context},
                        config={"configurable": {"session_id": "streamlit_chat_session"}},
                    )
                except Exception as e:
                    final_answer_with_suggestions = f"ขออภัยครับ เกิดข้อผิดพลาดในการเชื่อมต่อกับ Gemini: {e}"
                    st.error(final_answer_with_suggestions)

            # แสดงผลลัพธ์
            st.markdown(final_answer_with_suggestions)
            
            # แสดงขั้นตอนเบื้องหลัง (ดีสำหรับ Debug)
            with st.expander("ดูขั้นตอนการทำงานของ AI"):
                st.info(f"**คำค้นหาที่ AI สร้างขึ้นเพื่อค้นหาข้อมูล:**\n\n> {rewritten_question}")
                st.success(f"**ข้อมูลอ้างอิงที่พบและส่งให้ AI พิจารณา:**\n\n---\n{context}")

            # บันทึกผลลัพธ์ลง history
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer_with_suggestions
            })
            
            # Rerun เพื่อเคลียร์สถานะ (เป็นทางเลือก)
            # st.rerun()

else:
    st.warning("ระบบยังไม่พร้อมใช้งาน กรุณารอให้ Vector Store โหลดเสร็จสิ้น หรือตรวจสอบข้อผิดพลาดใน Terminal")