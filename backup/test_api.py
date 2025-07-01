import os
import requests
from dotenv import load_dotenv

print("🚀 เริ่มทดสอบการเชื่อมต่อ Typhoon API...")

# โหลด API Key จากไฟล์ .env
load_dotenv()
api_key = os.getenv("TAIFUUN_API_KEY")

if not api_key:
    print("❌ ไม่พบ TAIFUUN_API_KEY ในไฟล์ .env! กรุณาตรวจสอบ")
else:
    print("API Key: พบแล้ว")

    url = "https://api.aibuilder.in.th/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        # เราจะใช้ requests.options() ซึ่งเป็นการส่งคำขอที่เบาที่สุด
        # เพื่อเช็คว่าเรา "ไปถึง" เซิร์ฟเวอร์ได้หรือไม่ โดยไม่สนว่าคำขอจะถูกหรือผิด
        print(f"กำลังพยายามเชื่อมต่อไปยัง: {url}")
        response = requests.options(url, headers=headers, timeout=10) # ลองเชื่อมต่อภายใน 10 วินาที
        
        # ถ้าโค้ดมาถึงบรรทัดนี้ได้ แสดงว่าการเชื่อมต่อสำเร็จ
        print("\n" + "="*40)
        print("✅✅✅ การเชื่อมต่อสำเร็จ! ✅✅✅")
        print(f"สถานะที่ได้รับกลับมา: {response.status_code}")
        print("หมายความว่า Network ของน้องปกติดี สามารถคุยกับเซิร์ฟเวอร์ Typhoon ได้")
        print("="*40)

    except requests.exceptions.RequestException as e:
        # ถ้าเข้าตรงนี้ แสดงว่าเชื่อมต่อไม่ได้จริงๆ
        print("\n" + "="*40)
        print("❌❌❌ เกิดปัญหาในการเชื่อมต่อ Network! ❌❌❌")
        print("Error ที่ได้รับคือ:", e)
        print("\nสาเหตุที่เป็นไปได้สูง:")
        print("1. Firewall/Antivirus ในเครื่องบล็อกการเชื่อมต่อ")
        print("2. ปัญหาอินเทอร์เน็ต หรือ DNS ของผู้ให้บริการ")
        print("="*40)