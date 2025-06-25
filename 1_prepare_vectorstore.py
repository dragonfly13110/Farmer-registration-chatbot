# 1_prepare_vectorstore.py (ฉบับสมบูรณ์สำหรับ 2 ไฟล์: FAQ + Knowledge Base)

import os
import re
import shutil

# --- ส่วนที่ต้องใช้จาก LangChain ---
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- ตั้งค่า ---
# ตรวจสอบให้แน่ใจว่าไฟล์ Word ทั้งสองไฟล์อยู่ในโฟลเดอร์ data และชื่อตรงกัน
FAQ_DOCX_PATH = "data/farmer_guide.docx"  # ไฟล์ Q&A 100+ ข้อ
KB_DOCX_PATH = "data/knowledge_base.docx" # ไฟล์สรุปเนื้อหา 10 ช่วง
VECTORSTORE_PATH = "vectorstore_combined" # ชื่อของ Vector Store ที่จะถูกสร้าง
# เรายังคงใช้ Embedding Model ที่ดีที่สุดสำหรับภาษาไทย
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

def create_qa_chunks(text: str) -> list[str]:
    """
    ฟังก์ชันพิเศษสำหรับตัดแบ่งข้อความตามรูปแบบ Q&A (เช่น "1. ถาม:")
    ใช้ Regular Expression เพื่อหาและตัดข้อความที่ขึ้นต้นด้วย "ตัวเลข. ถาม:"
    """
    # Pattern นี้จะมองหา "ตัวเลข. ถาม:" และจับคู่ข้อความทั้งหมดไปจนถึงก่อน "ตัวเลข. ถาม:" อันถัดไป หรือจนจบไฟล์
    pattern = re.compile(r"(\d{1,3}\.\s*ถาม:.*?(?=\n\d{1,3}\.\s*ถาม:|\Z))", re.DOTALL)
    
    qa_pairs = pattern.findall(text)
    
    # ทำความสะอาดข้อความแต่ละชิ้น ตัดช่องว่างที่ไม่จำเป็นออก
    cleaned_pairs = [pair.strip() for pair in qa_pairs if pair.strip()]
    
    return cleaned_pairs

def main():
    """
    ฟังก์ชันหลักในการสร้าง Vector Store
    """
    print("🚀 เริ่มต้นสร้าง Vector Store แบบผสม (Combined)...")
    all_documents = []

    # --- ส่วนที่ 1: ประมวลผลไฟล์ FAQ (Q&A) ---
    if os.path.exists(FAQ_DOCX_PATH):
        print(f"กำลังประมวลผลไฟล์ FAQ: {FAQ_DOCX_PATH}")
        try:
            faq_loader = Docx2txtLoader(FAQ_DOCX_PATH)
            faq_text = faq_loader.load()[0].page_content
            faq_chunks = create_qa_chunks(faq_text)
            
            # ใส่ Metadata บอกว่ามาจากไฟล์ FAQ
            faq_docs = [Document(page_content=chunk, metadata={"source": "faq"}) for chunk in faq_chunks]
            all_documents.extend(faq_docs)
            print(f"✅ ประมวลผล FAQ สำเร็จ, พบ {len(faq_docs)} คู่ Q&A")
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการประมวลผลไฟล์ FAQ: {e}")
    else:
        print(f"⚠️ ไม่พบไฟล์ FAQ ที่: {FAQ_DOCX_PATH}")

    # --- ส่วนที่ 2: ประมวลผลไฟล์ Knowledge Base (เนื้อหาบรรยาย) ---
    if os.path.exists(KB_DOCX_PATH):
        print(f"กำลังประมวลผลไฟล์ Knowledge Base: {KB_DOCX_PATH}")
        try:
            kb_loader = Docx2txtLoader(KB_DOCX_PATH)
            kb_documents = kb_loader.load()
            # ใช้การตัดแบ่งแบบ Recursive สำหรับเนื้อหาบรรยาย
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
            kb_chunks = text_splitter.split_documents(kb_documents)
            
            # ใส่ Metadata บอกว่ามาจากไฟล์ KB
            for chunk in kb_chunks:
                chunk.metadata["source"] = "knowledge_base"
            all_documents.extend(kb_chunks)
            print(f"✅ ประมวลผล Knowledge Base สำเร็จ, แบ่งได้ {len(kb_chunks)} chunks")
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการประมวลผลไฟล์ Knowledge Base: {e}")
    else:
        print(f"⚠️ ไม่พบไฟล์ Knowledge Base ที่: {KB_DOCX_PATH}")

    # --- ส่วนที่ 3: สร้าง Vector Store ---
    if not all_documents:
        print("❌ ไม่พบข้อมูลจากไฟล์ใดๆ เลย! หยุดการทำงาน")
        return

    print(f"\nกำลังสร้าง Vector Store จากเอกสารทั้งหมด {len(all_documents)} ชิ้น...")
    
    # โหลด Embedding Model
    print(f"กำลังโหลด Embedding Model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # ลบ Vector Store เก่าทิ้งก่อนเสมอ เพื่อให้แน่ใจว่าเป็นข้อมูลล่าสุด
    if os.path.exists(VECTORSTORE_PATH):
        print(f"กำลังลบ Vector Store เก่าที่ '{VECTORSTORE_PATH}'...")
        shutil.rmtree(VECTORSTORE_PATH)
        print("✅ ลบสำเร็จ!")

    # สร้าง Vector Store ใหม่จากเอกสารทั้งหมด
    db = FAISS.from_documents(all_documents, embeddings)
    db.save_local(VECTORSTORE_PATH)
    print(f"✅ สร้าง Vector Store แบบผสมเสร็จสิ้น! บันทึกไว้ที่: {VECTORSTORE_PATH}")

if __name__ == "__main__":
    main()