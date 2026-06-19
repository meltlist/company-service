"""文档处理服务：PDF/Word/Excel 文本提取 + 智能分块"""
import hashlib
import re
import uuid
from pathlib import Path
from typing import Any

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    import openpyxl
except ImportError:
    openpyxl = None

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None


class DocumentProcessor:
    """文档处理器"""

    def __init__(self):
        self.ocr_engine = RapidOCR() if RapidOCR else None

    def extract_from_pdf(self, file_path: str) -> dict[str, Any]:
        """从 PDF 提取文本、标题结构、图片"""
        if not fitz:
            return {"text": "", "pages": [], "metadata": {}}

        doc = fitz.open(file_path)
        pages = []
        full_text = []

        for page_num, page in enumerate(doc, start=1):
            # 提取文本
            text = page.get_text("text")
            blocks = page.get_text("blocks")

            # 识别标题（根据字体大小）
            headings = self._extract_headings(blocks)

            # 提取图片（用于 OCR）
            images = page.get_images(full=True)
            image_texts = []
            for img in images:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                if self.ocr_engine:
                    result, _, _ = self.ocr_engine(image_bytes)
                    if result:
                        image_texts.append(" ".join([item[1] for item in result]))

            page_content = {
                "page_num": page_num,
                "text": text,
                "headings": headings,
                "images_text": " ".join(image_texts),
            }
            pages.append(page_content)
            full_text.append(text)

        doc.close()

        return {
            "text": "\n\n".join(full_text),
            "pages": pages,
            "metadata": {
                "page_count": len(pages),
                "title": self._extract_pdf_title(pages),
            },
        }

    def extract_from_docx(self, file_path: str) -> dict[str, Any]:
        """从 Word 文档提取文本和标题"""
        if not DocxDocument:
            return {"text": "", "pages": [], "metadata": {}}

        doc = DocxDocument(file_path)
        paragraphs = []
        full_text = []
        headings = []
        current_heading = None

        for para in doc.paragraphs:
            style_name = para.style.name if para.style else ""
            text = para.text.strip()

            if not text:
                continue

            # 判断标题级别
            if style_name.startswith("Heading"):
                level = int(style_name.replace("Heading ", "")) if "Heading " in style_name else 1
                headings.append({"level": level, "text": text})
                current_heading = text
            else:
                paragraphs.append({
                    "text": text,
                    "heading": current_heading,
                })
                full_text.append(text)

        # 处理表格
        tables_text = []
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells]
                if any(row_text):
                    tables_text.append(" | ".join(row_text))
            tables_text.append("")  # 表格之间空行

        return {
            "text": "\n".join(full_text) + "\n\n" + "\n".join(tables_text),
            "paragraphs": paragraphs,
            "metadata": {
                "headings": headings,
                "title": headings[0]["text"] if headings else "",
            },
        }

    def extract_from_excel(self, file_path: str) -> dict[str, Any]:
        """从 Excel 提取文本"""
        if not openpyxl:
            return {"text": "", "pages": [], "metadata": {}}

        wb = openpyxl.load_workbook(file_path, data_only=True)
        sheets_text = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            sheet_lines = [f"[Sheet: {sheet_name}]"]
            for row in ws.iter_rows(values_only=True):
                row_text = [str(cell) if cell is not None else "" for cell in row]
                if any(cell.strip() for cell in row_text):
                    sheet_lines.append(" | ".join(row_text))
            sheets_text.append("\n".join(sheet_lines))

        wb.close()

        return {
            "text": "\n\n".join(sheets_text),
            "metadata": {"sheets": wb.sheetnames},
        }

    def extract(self, file_path: str, file_ext: str) -> dict[str, Any]:
        """统一提取接口"""
        ext = file_ext.lower().replace(".", "")
        if ext in ("pdf",):
            return self.extract_from_pdf(file_path)
        elif ext in ("docx", "doc"):
            return self.extract_from_docx(file_path)
        elif ext in ("xlsx", "xls"):
            return self.extract_from_excel(file_path)
        else:
            return {"text": "", "metadata": {}}

    def _extract_headings(self, blocks: list) -> list[dict]:
        """从 PDF 块中识别标题"""
        headings = []
        for block in blocks:
            if len(block) >= 5:
                text = block[4].strip()
                # 根据字体大小判断（简化版）
                # 实际应用中需要根据具体 PDF 调整
                if text and len(text) < 100 and text.isupper():
                    headings.append({"text": text, "level": 1})
        return headings

    def _extract_pdf_title(self, pages: list) -> str:
        """提取 PDF 标题"""
        if pages and pages[0].get("headings"):
            return pages[0]["headings"][0].get("text", "")
        return ""


class TextChunker:
    """智能文本分块"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_by_semantic(self, text: str, metadata: dict = None) -> list[dict]:
        """基于语义的分块"""
        # 1. 先按段落分割
        paragraphs = self._split_paragraphs(text)

        # 2. 按语义合并小段落
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            # 如果单个段落超过最大块大小，按句子拆分
            if para_size > self.chunk_size * 1.5:
                sentences = self._split_sentences(para)
                for sentence in sentences:
                    if current_size + len(sentence) > self.chunk_size and current_chunk:
                        chunks.append(self._make_chunk(current_chunk))
                        current_chunk = current_chunk[-1:] if self.overlap > 0 else []
                        current_size = sum(len(p) for p in current_chunk)
                    current_chunk.append(sentence)
                    current_size += len(sentence)
            elif current_size + para_size > self.chunk_size:
                chunks.append(self._make_chunk(current_chunk))
                # 保留重叠部分
                overlap_size = 0
                current_chunk = []
                for p in reversed(current_chunk):
                    if overlap_size + len(p) <= self.overlap:
                        current_chunk.insert(0, p)
                        overlap_size += len(p)
                    else:
                        break
                current_size = overlap_size
                current_chunk.append(para)
                current_size += para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        if current_chunk:
            chunks.append(self._make_chunk(current_chunk))

        # 添加元数据
        for i, chunk in enumerate(chunks):
            chunk["metadata"] = {**(metadata or {}), "chunk_index": i}

        return chunks

    def chunk_by_page(self, pages: list, metadata: dict = None) -> list[dict]:
        """按页面分块"""
        chunks = []
        for page in pages:
            text = page.get("text", "")
            if text.strip():
                chunks.append({
                    "content": text.strip(),
                    "page_number": page.get("page_num"),
                    "chunk_type": "text",
                    "metadata": {
                        **(metadata or {}),
                        "chunk_index": len(chunks),
                    },
                })
        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """按段落分割"""
        # 清理多余空白
        text = re.sub(r"\n{3,}", "\n\n", text)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs

    def _split_sentences(self, text: str) -> list[str]:
        """按句子分割"""
        # 简单按句号、问号、感叹号分割
        sentences = re.split(r"(?<=[。！？.?!])", text)
        return [s.strip() for s in sentences if s.strip()]

    def _make_chunk(self, paragraphs: list[str]) -> dict:
        """生成块"""
        return {
            "content": "\n\n".join(paragraphs),
            "chunk_type": "text",
            "metadata": {},
        }


def compute_file_hash(file_path: str) -> str:
    """计算文件 MD5 哈希"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
