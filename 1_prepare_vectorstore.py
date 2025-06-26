import os
import re
import shutil

from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document

# --- การตั้งค่า ---
KB_DOCX_PATH = "data/knowledge_base.docx" # ไฟล์ฐานความรู้ฉบับละเอียดที่คุณเรียบเรียง
VECTORSTORE_PATH = "vectorstore_smart_chunking" # ชื่อ Vector Store ใหม่
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# --- ฟังก์ชัน Chunking แบบต่างๆ ---

def chunk_by_headers(text: str) -> list[Document]:
    """
    กลยุทธ์ที่ 1: ตัดแบ่งตามหัวข้อ Markdown (เหมาะกับเนื้อหาบรรยาย, กฎเกณฑ์)
    ใช้ #, ##, ### เป็นตัวแบ่ง จะทำให้เนื้อหาที่อยู่ใต้หัวข้อเดียวกันไม่แยกจากกัน
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # คงหัวข้อไว้ในเนื้อหาเพื่อเป็น context
    )
    docs = markdown_splitter.split_text(text)
    # เพิ่มการซอยย่อยสำหรับส่วนที่ยาวเกินไป
    final_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    for doc in docs:
        if len(doc.page_content) > 800:
            sub_docs = text_splitter.create_documents([doc.page_content], metadatas=[doc.metadata])
            final_docs.extend(sub_docs)
        else:
            final_docs.append(doc)
    return final_docs

def chunk_definitions(text: str) -> list[Document]:
    """
    กลยุทธ์ที่ 2: ตัดแบ่งนิยามศัพท์ (แต่ละนิยามคือ 1 chunk)
    ใช้ Regular Expression เพื่อหาหัวข้อที่มีตัวเลขนำหน้า (เช่น "1. เกษตรกร")
    """
    # มองหา Pattern "#### ตัวเลข. ชื่อเรื่อง" และจับคู่ข้อความทั้งหมดจนถึงหัวข้อถัดไป
    pattern = re.compile(r"(####\s*\d+\..*?(?=\n####\s*\d+\.|\Z))", re.DOTALL)
    definitions = pattern.findall(text)
    
    # ทำความสะอาดและสร้างเป็น Document
    docs = [Document(page_content=d.strip()) for d in definitions if d.strip()]
    return docs

def chunk_table_like_data(text: str, chunk_prefix: str) -> list[Document]:
    """
    กลยุทธ์ที่ 3: ตัดแบ่งข้อมูลแบบตาราง (แต่ละแถว/รายการ คือ 1 chunk)
    ใช้การ split by newline และเติม Prefix เพื่อให้ context ชัดเจน
    """
    # ลบหัวข้อหลักออกก่อน
    lines = text.strip().split('\n')
    header = lines[0] # เก็บหัวข้อหลักไว้
    items = [line.strip() for line in lines[2:] if line.strip().startswith('*')] # เอาเฉพาะบรรทัดที่เป็นรายการ

    docs = []
    for item in items:
        # สร้างเนื้อหา chunk โดยรวมหัวข้อหลักและรายการย่อย
        # นำเครื่องหมาย * และตัวหนาออก
        clean_item = item.replace('*', '').replace('**', '').strip()
        page_content = f"{chunk_prefix}: {clean_item}"
        docs.append(Document(page_content=page_content))
    return docs

def main():
    print("🚀 เริ่มต้นสร้าง Vector Store แบบ Smart Chunking...")
    
    if not os.path.exists(KB_DOCX_PATH):
        print(f"❌ ไม่พบไฟล์ฐานข้อมูลที่: {KB_DOCX_PATH}")
        return

    # --- 1. โหลดและแยกส่วนเนื้อหาตามตัวคั่นพิเศษ ---
    print(f"กำลังโหลดและแยกส่วนเนื้อหาจาก: {KB_DOCX_PATH}")
    loader = Docx2txtLoader(KB_DOCX_PATH)
    full_text = loader.load()[0].page_content
    
    # ใช้ Regex เพื่อแยกส่วนตาม ---[SECTION:NAME]---
    # (?s) คือ re.DOTALL, (.*?) คือ non-greedy match
    sections = re.findall(r"---SECTION:(.*?)---\n(?s)(.*?)(?=---SECTION:|$)", full_text)
    section_map = {name.strip(): content.strip() for name, content in sections}
    
    if not section_map:
        print("⚠️ ไม่พบตัวคั่น '---[SECTION:...]--' ในไฟล์! จะใช้การตัดแบ่งแบบ Recursive ทั้งหมดแทน")
        # Fallback to simple recursive splitter if no sections found
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        all_documents = text_splitter.create_documents([full_text])
    else:
        # --- 2. ใช้กลยุทธ์ Chunking ที่ต่างกันในแต่ละส่วน ---
        print("พบเนื้อหา {len(section_map)} ส่วน! กำลังใช้กลยุทธ์ตัดแบ่งที่แตกต่างกัน...")
        all_documents = []

        if "DEFINITIONS" in section_map:
            print("  - 📝 กำลังประมวลผล 'DEFINITIONS' แบบ chunk-per-definition...")
            docs = chunk_definitions(section_map["DEFINITIONS"])
            for doc in docs: doc.metadata["source"] = "definitions"
            all_documents.extend(docs)
            print(f"    -> สร้างได้ {len(docs)} chunks")

        if "RULES" in section_map:
            print("  - 📜 กำลังประมวลผล 'RULES' แบบ chunk-by-header...")
            docs = chunk_by_headers(section_map["RULES"])
            for doc in docs: doc.metadata["source"] = "rules_and_conditions"
            all_documents.extend(docs)
            print(f"    -> สร้างได้ {len(docs)} chunks")
            
        if "HOW_TO_GUIDE" in section_map:
            print("  - 👣 กำลังประมวลผล 'HOW_TO_GUIDE' แบบ chunk-by-header...")
            docs = chunk_by_headers(section_map["HOW_TO_GUIDE"])
            for doc in docs: doc.metadata["source"] = "how_to_guide"
            all_documents.extend(docs)
            print(f"    -> สร้างได้ {len(docs)} chunks")
            
        if "TIMELINES" in section_map:
            print("  - ⏰ กำลังประมวลผล 'TIMELINES' แบบ chunk-by-header...")
            docs = chunk_by_headers(section_map["TIMELINES"])
            for doc in docs: doc.metadata["source"] = "timelines"
            all_documents.extend(docs)
            print(f"    -> สร้างได้ {len(docs)} chunks")

        if "PLANTING_DENSITY" in section_map:
            print("  - 🌳 กำลังประมวลผล 'PLANTING_DENSITY' แบบ chunk-per-item...")
            docs = chunk_table_like_data(
                section_map["PLANTING_DENSITY"],
                chunk_prefix="เกณฑ์จำนวนต้นต่อไร่"
            )
            for doc in docs: doc.metadata["source"] = "planting_density"
            all_documents.extend(docs)
            print(f"    -> สร้างได้ {len(docs)} chunks")

    if not all_documents:
        print("❌ ไม่สามารถสร้างเอกสารใดๆ ได้! หยุดการทำงาน")
        return

    # --- 3. สร้าง Vector Store ---
    print(f"\nกำลังสร้าง Vector Store จากเอกสารทั้งหมด {len(all_documents)} ชิ้น...")
    
    print(f"กำลังโหลด Embedding Model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cuda'}) # Use 'cpu' if no GPU
    
    if os.path.exists(VECTORSTORE_PATH):
        print(f"กำลังลบ Vector Store เก่าที่ '{VECTORSTORE_PATH}'...")
        shutil.rmtree(VECTORSTORE_PATH)

    db = FAISS.from_documents(all_documents, embeddings)
    db.save_local(VECTORSTORE_PATH)
    print(f"✅ สร้าง Vector Store แบบ Smart Chunking เสร็จสิ้น! บันทึกไว้ที่: {VECTORSTORE_PATH}")


if __name__ == "__main__":
    main()