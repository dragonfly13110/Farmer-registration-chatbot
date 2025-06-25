# 2_inspect_chunks.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- ตั้งค่า ---
PDF_PATH = "data/farmer_guide.pdf"

print(f"กำลังตรวจสอบการตัดแบ่งเอกสารจาก: {PDF_PATH}")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# ลองใช้ค่า chunking เดิมก่อน
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

print("\n" + "="*50)
print(f"ตัดเอกสารได้ทั้งหมด {len(texts)} ชิ้น (chunk)")
print("นี่คือตัวอย่าง 5 ชิ้นแรก:")
print("="*50 + "\n")

for i, text in enumerate(texts[:5]): # แสดง 5 ชิ้นแรก
    print(f"--- ชิ้นที่ {i+1} ---")
    print(text.page_content)
    print("\n" + "-"*20 + "\n")