import os
import glob
import fitz
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from google import genai

app = FastAPI(title="API Tro ly Luu tru Can Tho")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

chunks = []
bm25 = None

def load_abbyy_pdfs(folder_path="./tai_lieu_pdf"):
    global chunks, bm25
    chunks = []
    tokenized_corpus = []
    
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    for file_path in pdf_files:
        file_name = os.path.basename(file_path)
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text").strip()
            if len(text) > 30:
                chunks.append({
                    "file": file_name,
                    "page": page_num + 1,
                    "content": text
                })
                tokenized_corpus.append(text.lower().split())
                
    if tokenized_corpus:
        bm25 = BM25Okapi(tokenized_corpus)
    print(f"Da nap {len(chunks)} trang tu {len(pdf_files)} file PDF.")

load_abbyy_pdfs()

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def health_check():
    return {"status": "online", "total_pages_indexed": len(chunks)}

@app.post("/ask")
def ask_archive(req: ChatRequest):
    if not chunks or bm25 is None:
        return {"answer": "Hệ thống chưa có tài liệu nào được nạp vào thư mục dữ liệu."}

    tokenized_query = req.question.lower().split()
    top_n = bm25.get_top_n(tokenized_query, chunks, n=4)

    context_text = ""
    for item in top_n:
        context_text += f"\n--- Nguồn: File '{item['file']}' (Trang {item['page']}) ---\n{item['content']}\n"

    prompt = f"""Bạn là Trợ lý ảo của Trung tâm Lưu trữ lịch sử thành phố Cần Thơ.
Dưới đây là nội dung trích xuất từ kho tài liệu lưu trữ đã qua xử lý OCR:
{context_text}

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin được cung cấp ở trên để trả lời câu hỏi của người dân. Tuyệt đối không tự suy diễn ngoài tài liệu.
2. Trình bày rõ ràng, lịch sự, chuẩn mực hành chính.
3. Cuối câu trả lời, luôn ghi rõ trích dẫn tên file và số trang để người dân làm thủ tục mượn bản gốc tại Phòng Đọc.
4. Nếu thông tin không có trong tài liệu trên, hãy trả lời: "Hiện tại hệ thống trực tuyến chưa tìm thấy tài liệu này. Quý khách vui lòng liên hệ trực tiếp Phòng Đọc của Trung tâm để được hỗ trợ tra cứu chuyên sâu."

Câu hỏi của người dân: {req.question}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return {"answer": response.text}
    except Exception as e:
        return {"answer": f"Hệ thống đang bận, vui lòng thử lại sau. (Lỗi kỹ thuật: {str(e)})"}
