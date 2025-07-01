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
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

# --- โหลดค่าตั้งค่าและโมเดล ---
load_dotenv()
api_keys = [key for key in os.environ.keys() if key.startswith("GOOGLE_API_KEY")]
api_key_pool = [os.getenv(key) for key in api_keys if os.getenv(key)]

if not api_key_pool:
    st.error("ไม่พบ Google API Key ใดๆ ในไฟล์ .env! กรุณาตรวจสอบ")
    st.stop()

VECTORSTORE_PATH = "vectorstore_smart_chunking_v2" 
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
LLM_MODEL = "gemini-2.5-flash" 

# [เพิ่ม] กำหนดค่าคงที่สำหรับขนาดของ history เพื่อให้ง่ายต่อการปรับแก้..
MAX_HISTORY_MESSAGES = 6

# --- ฟังก์ชันหลัก (Cached) ---

@st.cache_resource
def load_vector_store():
    """โหลด Vector Store ที่สร้างไว้แล้ว"""
    if not os.path.exists(VECTORSTORE_PATH):
        st.error(f"ไม่พบฐานข้อมูล Vector Store ที่ '{VECTORSTORE_PATH}'! กรุณารันไฟล์ prepare_vectorstore.py ก่อน")
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cuda' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu'})
        db = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลด Vector Store: {e}")
        return None

store = {}
# [แก้ไข] ปรับปรุงฟังก์ชันนี้เพื่อจำกัดขนาดของ history (Sliding Window)
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    ดึงประวัติการแชตสำหรับ session ปัจจุบัน และจัดการขนาดของ history
    เพื่อให้ไม่เกินจำนวนที่กำหนด
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    
    # ดึง history object ทั้งหมดออกมาจาก store
    session_history = store[session_id]
    
    # ตรวจสอบขนาดของ history และตัดให้เหลือแค่ k ข้อความล่าสุด
    # การทำแบบนี้จะแก้ไขที่ตัว object ใน store โดยตรง ทำให้ history ไม่สะสมยาวเกินไป
    if len(session_history.messages) > MAX_HISTORY_MESSAGES:
        session_history.messages = session_history.messages[-MAX_HISTORY_MESSAGES:]
        
    return session_history

def format_docs(docs: list[Document]) -> str:
    """จัดรูปแบบเอกสารที่ค้นเจอให้เป็นข้อความเดียวเพื่อง่ายต่อการอ่านของ LLM"""
    formatted_docs = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "ไม่ระบุ")
        content = doc.page_content
        formatted_docs.append(f"เอกสารอ้างอิงชิ้นที่ {i+1} (ที่มา: {source}):\n{content}")
    return "\n\n---\n\n".join(formatted_docs)

@st.cache_resource
def get_chains(_retriever):
    """สร้าง Chains ทั้งหมด (Rewriter + RAG)"""
    st.write("กำลังเตรียมผู้ช่วย AI...")
    
    selected_key = random.choice(api_key_pool)
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.7, # ลด Temp ลงเพื่อความแม่นยำ
        google_api_key=selected_key,
    )

    # --- Chain 1: สำหรับแปลงคำถาม (Rewriter) ---
    rewriter_prompt = ChatPromptTemplate.from_messages([
        ("system", """คุณคือผู้ช่วย AI ที่เชี่ยวชาญในการแปลงคำถามของผู้ใช้ให้เป็นคำค้นหา ที่มีประสิทธิภาพสำหรับ Vector Database
ภารกิจของคุณคือการอ่านบทสนทนาล่าสุดและคำถามติดตามผล แล้วสร้างคำค้นหาที่เป็นประโยคสมบูรณ์ ที่เหมาะสำหรับการค้นหาข้อมูลในคู่มือการขึ้นทะเบียนเกษตรกร
- แปลงภาษาพูดให้เป็นภาษาที่เป็นทางการมากขึ้น (เช่น "ทำสวน" -> "การเพาะปลูกพืช", "ต้องใช้อะไรบ้าง" -> "เอกสารและคุณสมบัติที่จำเป็น")
- ให้ตีความด้วย ชนิดพืชที่ผู้ถามกล่าวถึง มันหมายความว่าไง เข้าเกณธ์พืชประเภอไหน และต้องไปค้นหาหลักเกณฑ์อะไรบ้าง 
- รวมบริบทที่สำคัญจากบทสนทนาก่อนหน้าเข้ามาในคำค้นหาใหม่"""),
        
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    rewriter_chain = rewriter_prompt | llm | StrOutputParser()

    # --- Chain 2: สำหรับสร้างคำตอบ (RAG) ---
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """คุณคือ "ผู้เชี่ยวชาญในการขึ้นทะเบียน กรมส่งเสริมการเกษตร" ที่เป็นมิตร แม่นยำและรอบคอบและชัดเจน ที่สำคัญคือต้องกระชับ

**กฎเหล็กสูงสุด:**
1.  **ยึดข้อมูลอ้างอิงเท่านั้น:** ตอบคำถามโดยใช้ข้อมูลจาก "ข้อมูลอ้างอิง" (context) ที่ให้มา **เท่านั้น** และให้ตอบกระชับ ฟังธงไปเลย แต่ถ้าไม่มีในcontext จริงๆ ให้คุณใช้องค์ความรู้ของคุณประกอบ และให้แนะนำให้ผู้ใช้ติดต่อเจ้าหน้าที่ที่เกี่ยวข้อง
2.  **การให้แนะนำเรื่องข้อยกเว้นเรื่องรายได้ สำคัญมาก ซึ่งเกณฑ์นี้ใหญ่สุด ทะลุข้อจำกัดทุกอย่าง ทั้งเรื่องพื้นที่ไม่ถึงเกณฑ์ ทั้งเรื่องจำนวนต้นไม่ถึงเกณฑ์ อันนี้สำคัญมากๆ ห้ามลืมแนะนำเด็ดขาด**
3.  **ทุกครั้ง ถ้ามีการถามว่าปลูกสิ่งต่าง ๆ แล้วขึ้นทะเบียนได้ไหม ให้แนะนำเรื่องพื้นที่ปลูกขั้นต่อของพืชนั้นๆ และถ้ามีจำนวนต้นขั้นต่ำ/พื้นที่ ก็ให้แนะนำด้วย แต่ถ้าคุณพิจารณาคำถามแล้ว พื้นที่อาจจะไม่ถึงพื้นที่ขั้นต่ำ ให้แนะนำเรื่องข้อยกเว้นเรื่องรายได้ 
4.  **จัดลำดับความสำคัญ:** หากข้อมูลอ้างอิงมีเนื้อหาที่ดูขัดแย้งกัน ให้เชื่อถือข้อมูลจาก Chunk ที่มีเนื้อหา **ตรงกับคำถามของผู้ใช้มากที่สุดก่อนเสมอ** (เช่น หากถามถึง "ถั่วงอก" ให้เชื่อข้อมูลเกี่ยวกับ "การทำผักงอก" มากกว่าข้อมูล "พืชไร่" ทั่วไป)
5.  **ให้คำแนะนำแบบองค์รวม (Holistic Advice Rule):** หากคำถามของผู้ใช้มีกิจกรรมหลายอย่างปนกัน (ทั้งในและนอกขอบเขต) **ห้ามปฏิเสธแล้วจบ** แต่ให้ทำตามขั้นตอนต่อไปนี้:
    *   **แยกแยะ:** บอกผู้ใช้ว่ากิจกรรมไหน "ขึ้นทะเบียนได้" (กับกรมส่งเสริมการเกษตร) และกิจกรรมไหน "ขึ้นทะเบียนไม่ได้" (เพราะอยู่นอกขอบเขต)
    *   **ให้ข้อมูลส่วนที่ทำได้:** ให้ข้อมูลและคำแนะนำอย่างละเอียดสำหรับกิจกรรมที่ "ขึ้นทะเบียนได้" (เช่น สวนเงาะ)
    *   **แนะนำส่วนที่ทำไม่ได้:** แนะนำให้ผู้ใช้ไปติดต่อหน่วยงานที่ถูกต้องสำหรับกิจกรรมที่ "ขึ้นทะเบียนไม่ได้" (เช่น การเลี้ยงเป็ด ให้ติดต่อกรมปศุสัตว์)
    *   **ตัวอย่างคำตอบที่คาดหวัง:** "สำหรับการขึ้นทะเบียนเกษตรกรกับกรมส่งเสริมการเกษตรนั้น ผมขออนุญาตแยกเป็น 2 ส่วนนะครับ:
        1.  **ในส่วนของ "สวนเงาะ":** คุณสามารถนำมาขึ้นทะเบียนได้ครับ โดยจะต้องเข้าเกณฑ์... (ให้ข้อมูลของเงาะต่อไป)
        2.  **ในส่วนของ "การเลี้ยงเป็ด":** กิจกรรมนี้จัดเป็นปศุสัตว์ ซึ่งจะอยู่นอกขอบเขตของกรมส่งเสริมการเกษตรครับ แนะนำให้ลองติดต่อสอบถามที่สำนักงานปศุสัตว์อำเภอโดยตรงเพื่อขึ้นทะเบียนในส่วนนี้ครับ"
6.  **จัดรูปแบบคำตอบด้วย Bullet points หรือย่อหน้าสั้นๆ ตอบกระชับ เพื่อให้อ่านง่าย
7.  **ถ้าถามอะไรที่ไม่เกี่ยวข้องกับการขึ้นทะเบียนเกษตรกร:** เช่น "ร้านขายยางรถยนต์" หรือ "หวยจะออกอะไร" ให้วิเคราะห์ว่า "คำถามนี้ไม่เกี่ยวข้องกับการขึ้นทะเบียนเกษตรกร" และแนะนำให้สอบถามเรื่องอื่นที่เกี่ยวข้องกับการขึ้นทะเบียนเกษตร
8.  **ให้คุณวิเคราะห์ด้วยว่า ถ้าเป็นคำถามซับซ้อน ให้แนะนำตอนท้ายว่า ควรปรึกษากับเจ้าหน้าที่ที่เกี่ยวข้องโดยตรง เพื่อความถูกต้องและแม่นยำที่สุด 


---

**ภารกิจและขั้นตอนการทำงานของคุณ (เมื่อพบข้อมูลใน context):**
**ขั้นตอนที่ 1: วิเคราะห์และทำความเข้าใจ**
- อ่าน "คำถามของผู้ใช้" และ "ประวัติการสนทนา" เพื่อทำความเข้าใจเจตนาที่แท้จริง

**ขั้นตอนที่ 2: สังเคราะห์คำตอบจากข้อมูลอ้างอิง**
- สร้างคำตอบที่ กระชับ ชัดเจน และถูกต้อง 100% ตามข้อมูลที่พบ แต่ให้ครอบคลุมทุกประเด็นที่เกี่ยวข้องกัน
**ขั้นตอนที่ 3: สร้างคำตอบตามผลการวิเคราะห์ ให้ตอบตามข้อมูลที่พบใน Context หากข้อมูลไม่ครบถ้วน ค่อยแนะนำให้สอบถามเจ้าหน้าที่เพิ่มเติม
**ขั้นตอนที่ 4: สร้างแนวทางคำถามต่อไป**
- **หลังจาก** ตอบคำถามหลักเสร็จสิ้นแล้ว ให้เว้นบรรทัด 2 บรรทัด
- เริ่มต้นด้วยข้อความว่า "**💡 ลองถามต่อได้เลย:**"
- ตามด้วยรายการคำถามแนะนำ 2 ข้อ ที่สั้น กระชับ เน้นไปที่ การขึ้นทะเบียนเกษตรกร เช่น เอกสารที่ต้องใช้, ขั้นตอนการขึ้นทะเบียน, คุณสมบัติที่จำเป็น

----
**ข้อมูลอ้างอิง:**
{context}
----
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    # --- ประกอบร่าง RAG Chain ---
    rag_chain_with_source = RunnableParallel(
        standalone_question=rewriter_chain,
        original_input=RunnablePassthrough()
    ) | RunnableParallel(
        context=lambda x: format_docs(_retriever.invoke(x["standalone_question"])),
        question=lambda x: x["original_input"]["question"],
        chat_history=lambda x: x["original_input"]["chat_history"]
    )
    
    rag_chain_with_dict_output = rag_chain_with_source | RunnablePassthrough.assign(
        answer=answer_prompt | llm | StrOutputParser()
    )
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain_with_dict_output,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    return conversational_rag_chain

# --- UI และ Logic หลัก ---
st.set_page_config(page_title="เกษตรกรแชตบอท", page_icon="👩‍🌾", layout="wide")
st.title("👩‍🌾 แชตบอทถาม-ตอบเรื่องการขึ้นทะเบียนเกษตรกร")
st.write("ขับเคลื่อนโดย Google Gemini และคู่มือทะเบียนเกษตรกรปี 2568 ผลิตโดย เกษตรตำบล_คนใช้แรงงาน")

db = load_vector_store()

if db:
    # สร้าง Retriever ที่ใช้ MMR เพื่อผลการค้นหาที่ดีขึ้น
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 8, 'fetch_k': 25} # [ปรับคืนค่าเดิม]
    )

    rag_chain_with_history = get_chains(retriever)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [AIMessage(content="สวัสดีครับ มีเรื่องการขึ้นทะเบียนเกษตรกรอะไรให้ผมช่วยเหลือไหมครับ?")]

    for msg in st.session_state.messages:
        st.chat_message(msg.type).write(msg.content)

    if user_input := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):
        st.session_state.messages.append(HumanMessage(content=user_input))
        st.chat_message("human").write(user_input)

        with st.chat_message("ai"):
            with st.spinner("กำลังประมวลผล..."):
                try:
                    # เรียกใช้ RAG Chain ที่มีระบบจัดการ history ในตัว
                    response_dict = rag_chain_with_history.invoke(
                        {"question": user_input},
                        config={"configurable": {"session_id": "main_session"}}
                    )
                    # ดึงค่าจาก key 'answer' และ 'context' ที่ได้จาก Chain
                    final_answer = response_dict.get("answer", "ขออภัยครับ เกิดข้อผิดพลาดในการดึงคำตอบ")
                    retrieved_context = response_dict.get("context", "ไม่พบข้อมูลอ้างอิง")

                except Exception as e:
                    final_answer = f"ขออภัยครับ เกิดข้อผิดพลาด: {e}"
                    retrieved_context = f"เกิดข้อผิดพลาดระหว่างการดึงข้อมูล: {e}"
                    st.error(final_answer)

            st.write(final_answer)
            
            # [เพิ่ม] แสดง Expander พร้อมข้อมูลอ้างอิงที่ใช้
            with st.expander("ดูข้อมูลอ้างอิงที่ AI ใช้ในการตอบคำถามนี้"):
                if isinstance(retrieved_context, str):
                    st.info(retrieved_context)
                else:
                    st.json(retrieved_context) # Fallback to json if not a string

        st.session_state.messages.append(AIMessage(content=final_answer))

else:
    st.warning("ระบบยังไม่พร้อมใช้งาน กรุณารอให้ Vector Store โหลดเสร็จสิ้น หรือตรวจสอบข้อผิดพลาดใน Terminal")