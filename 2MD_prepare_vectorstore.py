import os
import re
import shutil

# นำเข้าไลบรารีที่จำเป็นจาก LangChain
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document

# --- การตั้งค่าหลัก ---
# ใช้ไฟล์ Markdown เป็นแหล่งข้อมูลหลัก
KB_MARKDOWN_PATH = "data/knowledge_base.md"
QNA_MARKDOWN_PATH = "data/Q&A.md" # เพิ่มไฟล์ Q&A.md
# ตั้งชื่อ Vector Store ให้สื่อถึงวิธีการสร้าง
VECTORSTORE_PATH = "vectorstore_smart_chunking_v2" 
# Embedding Model ที่ดีที่สุดสำหรับภาษาไทย
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# --- ฟังก์ชันสำหรับกลยุทธ์การตัดแบ่ง (Chunking Strategies) ---

def chunk_by_headers(text: str) -> list[Document]:
    """
    กลยุทธ์ที่ 1: ตัดแบ่งตามหัวข้อ Markdown (เหมาะกับเนื้อหาบรรยาย, กฎเกณฑ์, ขั้นตอน)
    ใช้ #, ##, ###, #### เป็นตัวแบ่ง ทำให้เนื้อหาใต้หัวข้อเดียวกันไม่แยกจากกัน
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False  # คงหัวข้อไว้ในเนื้อหาเพื่อเป็น context
    )
    docs = markdown_splitter.split_text(text)
    
    # เพิ่มการซอยย่อยสำหรับส่วนที่ยาวเกินไป เพื่อให้แต่ละ chunk ไม่ใหญ่เกิน
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
    ใช้ Regular Expression เพื่อหาหัวข้อที่มีตัวเลขนำหน้า (เช่น "#### **1. เกษตรกร**")
    """
    # [แก้ไข] ปรับปรุง Regex ให้รองรับ Markdown ตัวหนา (**) ได้
    pattern = re.compile(r"(####\s*(\*\*)*\d+\..*?(?=\n####\s*(\*\*)*\d+\.|\Z))", re.DOTALL)
    
    # findall กับ capturing group จะคืนค่าเป็น list ของ tuple, เราเอาเฉพาะส่วนที่ match ทั้งหมด (index 0)
    matches = pattern.findall(text)
    definitions = [match[0] for match in matches]
    
    # ทำความสะอาดและสร้างเป็น Document
    docs = [Document(page_content=d.strip()) for d in definitions if d.strip()]
    return docs

def chunk_table_like_data(text: str, chunk_prefix: str) -> list[Document]:
    """
    กลยุทธ์ที่ 3: ตัดแบ่งข้อมูลแบบรายการ/ตาราง (แต่ละรายการคือ 1 chunk)
    ใช้การ split by newline และเติม Prefix ของหัวข้อเข้าไปเพื่อเพิ่ม context
    """
    lines = text.strip().split('\n')
    # ค้นหารายการที่ขึ้นต้นด้วย '*'
    items = [line.strip() for line in lines if line.strip().startswith('*')]

    docs = []
    for item in items:
        # นำเครื่องหมาย * และ ** ออกเพื่อความสะอาด
        clean_item = item.replace('*', '', 1).replace('**', '').strip()
        page_content = f"{chunk_prefix}: {clean_item}"
        docs.append(Document(page_content=page_content))
    return docs

def parse_qna_markdown(text: str) -> list[Document]:
    """
    กลยุทธ์เฉพาะ: ตัดแบ่ง Q&A จากไฟล์ Q&A.md
    แต่ละคำถาม-คำตอบจะถูกรวมเป็น 1 chunk
    """
    qna_docs = []
    lines = text.split('\n')
    current_category = ""
    current_subcategory = ""
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Extract main category (## หมวด)
        if line.startswith('## หมวด'):
            current_category = line.replace('## หมวด', '').strip()
            current_subcategory = "" # Reset subcategory
            i += 1
            continue
        # Extract subcategory (### หมวด)
        elif line.startswith('### หมวด'):
            current_subcategory = line.replace('### หมวด', '').strip()
            i += 1
            continue
        
        # Check for question pattern: 1.  **ถาม:** "..."
        q_match = re.match(r'^\d+\.\s+\*\*ถาม:\*\*\s*(.*)$', line)
        if q_match:
            question_text = q_match.group(1).strip()
            # Remove leading/trailing quotes if they exist
            if question_text.startswith('"') and question_text.endswith('"'):
                question_text = question_text[1:-1]
            
            # Look for answer in the next line(s)
            answer_lines = []
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line.startswith('> **ตอบ:**'):
                    answer_lines.append(next_line.replace('> **ตอบ:**', '').strip())
                    j += 1
                    # Continue collecting blockquote lines if they are part of the same answer
                    while j < len(lines) and lines[j].strip().startswith('>'):
                        answer_lines.append(lines[j].strip().lstrip('>').strip())
                        j += 1
                    break # Found answer, break inner loop
                elif next_line.strip() == '---' or re.match(r'^\d+\.\s+\*\*ถาม:\*\*', next_line) or next_line.startswith('## หมวด') or next_line.startswith('### หมวด'):
                    break
                else:
                    j += 1
            
            answer = " ".join(answer_lines).strip()
            
            if question_text and answer:
                full_content = f"หมวด: {current_category}"
                if current_subcategory:
                    full_content += f" / {current_subcategory}"
                full_content += f"\nคำถาม: {question_text}\nคำตอบ: {answer}"
                
                metadata = {
                    "source": "Q&A", 
                    "category": current_category
                }
                if current_subcategory:
                    metadata["subcategory"] = current_subcategory
                
                qna_docs.append(Document(page_content=full_content, metadata=metadata))
            
            i = j # Move index to after the answer
        else:
            i += 1 # Move to next line if not a question

    return qna_docs

def main():
    """ฟังก์ชันหลักในการสร้าง Vector Store"""
    print("🚀 เริ่มต้นสร้าง Vector Store แบบ Smart Chunking...")
    
    if not os.path.exists(KB_MARKDOWN_PATH):
        print(f"❌ ไม่พบไฟล์ฐานข้อมูลที่: {KB_MARKDOWN_PATH}")
        return
    if not os.path.exists(QNA_MARKDOWN_PATH):
        print(f"❌ ไม่พบไฟล์ Q&A ที่: {QNA_MARKDOWN_PATH}")
        return

    all_documents = []

    # --- 1. โหลดและแยกส่วนเนื้อหาจาก knowledge_base.md ---
    print(f"กำลังโหลดและแยกส่วนเนื้อหาจาก: {KB_MARKDOWN_PATH}")
    loader_kb = TextLoader(KB_MARKDOWN_PATH, encoding="utf-8")
    full_text_kb = loader_kb.load()[0].page_content
    
    # ใช้ re.split เพื่อแยกส่วนตามตัวคั่น ---[SECTION:NAME]---
    section_delimiter_pattern = r'---\[SECTION:(.*?)\]---'
    parts_kb = re.split(section_delimiter_pattern, full_text_kb)

    # จัดการผลลัพธ์จาก re.split
    # ผลลัพธ์จะเป็น [เนื้อหาก่อนตัวคั่นแรก, ชื่อsection1, เนื้อหาsection1, ชื่อsection2, เนื้อหาsection2, ...]
    section_map_kb = {}
    if len(parts_kb) > 1:
        # เราไม่เอาส่วนแรก (index 0) และจะจับคู่ ชื่อ กับ เนื้อหา
        section_names_kb = parts_kb[1::2]
        section_contents_kb = parts_kb[2::2]
        section_map_kb = {name.strip(): content.strip() for name, content in zip(section_names_kb, section_contents_kb)}

    if not section_map_kb:
        print("⚠️ ไม่พบตัวคั่น '---[SECTION:...]--' ในไฟล์ knowledge_base.md! จะใช้การตัดแบ่งแบบ Recursive ทั้งหมดแทน")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        kb_documents = text_splitter.create_documents([full_text_kb])
        all_documents.extend(kb_documents)
    else:
        # --- 2. ใช้กลยุทธ์ Chunking ที่แตกต่างกันในแต่ละส่วน ---
        print(f"พบเนื้อหา {len(section_map_kb)} ส่วนใน knowledge_base.md! กำลังใช้กลยุทธ์ตัดแบ่งที่แตกต่างกัน...")
        
        # สร้าง mapping ระหว่างชื่อ Section และฟังก์ชันที่ใช้
        strategy_map = {
            "DEFINITIONS": ("📝 ...", chunk_definitions, {}),
            "RULES": ("📜 ...", chunk_by_headers, {}),
            "HOW_TO_GUIDE": ("👣 ...", chunk_by_headers, {}),
            "MAINTENANCE": ("⚙️ ...", chunk_by_headers, {}),
            "TIMELINES": ("⏰ ...", chunk_by_headers, {}),
            "PLANTING_DENSITY": ("🌳 ...", chunk_table_like_data, {"chunk_prefix": "เกณฑ์จำนวนต้นต่อไร่"}),
            "MINIMUM_AREA": ("📏 กำลังประมวลผล 'MINIMUM_AREA' แบบ chunk-by-header...", chunk_by_headers, {}) 
        }

        for name, content in section_map_kb.items():
            if name in strategy_map:
                message, chunk_func, kwargs = strategy_map[name]
                print(f"  - {message}")
                docs = chunk_func(content, **kwargs)
                for doc in docs:
                    doc.metadata["source"] = name.lower() # ใส่ metadata บอกแหล่งที่มา
                all_documents.extend(docs)
                print(f"    -> สร้างได้ {len(docs)} chunks")
            else:
                print(f"  - ❓ ไม่พบกลยุทธ์สำหรับ '{name}' ใน knowledge_base.md, จะใช้การตัดแบ่งแบบ Recursive แทน...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
                docs = text_splitter.create_documents([content])
                all_documents.extend(docs)

    # --- 2. โหลดและแยกส่วนเนื้อหาจาก Q&A.md ---
    print(f"\nกำลังโหลดและแยกส่วนเนื้อหาจาก: {QNA_MARKDOWN_PATH}")
    loader_qna = TextLoader(QNA_MARKDOWN_PATH, encoding="utf-8")
    full_text_qna = loader_qna.load()[0].page_content
    
    qna_documents = parse_qna_markdown(full_text_qna)
    all_documents.extend(qna_documents)
    print(f"  -> สร้างได้ {len(qna_documents)} Q&A chunks")


    if not all_documents:
        print("❌ ไม่สามารถสร้างเอกสารใดๆ ได้! หยุดการทำงาน")
        return

    # --- 3. สร้าง Vector Store ---
    print(f"\nกำลังสร้าง Vector Store จากเอกสารทั้งหมด {len(all_documents)} ชิ้น...")
    
    print(f"กำลังโหลด Embedding Model: {EMBEDDING_MODEL}")
    # ใช้ GPU ถ้ามี, ถ้าไม่มีจะใช้ CPU อัตโนมัติ
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, 
        model_kwargs={'device': 'cuda' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu'}
    )
    
    if os.path.exists(VECTORSTORE_PATH):
        print(f"กำลังลบ Vector Store เก่าที่ '{VECTORSTORE_PATH}'...")
        shutil.rmtree(VECTORSTORE_PATH)

    db = FAISS.from_documents(all_documents, embeddings)
    db.save_local(VECTORSTORE_PATH)
    print(f"✅ สร้าง Vector Store (Smart Chunking v2) เสร็จสิ้น! บันทึกไว้ที่: {VECTORSTORE_PATH}")

if __name__ == "__main__":
    main()