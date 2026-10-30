"""文件解析与 OCR：把 PDF / 图片转成可检索文本，并切分为文本块。

PDF 后端可切换（PDF_BACKEND=auto|pypdf|pymupdf，默认 auto）：
- pypdf：纯 Python，负责文本抽取 / 合并 / 旋转；
- PyMuPDF(fitz)：渲染页面为图片（扫描页 OCR 需要），部分安全策略会拦截其 DLL，因此作为可选增强。
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import config

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_SUFFIXES = {".pdf"}
TEXT_SUFFIXES = {".txt", ".md"}


@dataclass
class ParseResult:
    text: str = ""
    page_count: int = 0
    ocr_pages: int = 0
    source_type: str = ""            # pdf-text | pdf-ocr | image-ocr | text
    status: str = "ok"               # ok | empty | error
    message: str = ""
    warnings: list[str] = field(default_factory=list)


def clean_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------- PDF 后端 ----------------
def fitz_available() -> bool:
    if config.PDF_BACKEND == "pypdf":
        return False
    try:
        import fitz  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - 安全策略可能拦截 DLL
        return False


class PdfRenderer:
    """把 PDF 某页渲染成 PIL 图片；优先 PyMuPDF，缺失时尝试 pypdfium2，再不行用 pypdf 提取内嵌图片。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fitz_doc = None
        self._pdfium_doc = None
        if fitz_available():
            import fitz

            self._fitz_doc = fitz.open(path)
            return
        try:
            import pypdfium2 as pdfium

            self._pdfium_doc = pdfium.PdfDocument(str(path))
        except Exception:  # noqa: BLE001
            self._pdfium_doc = None

    @property
    def page_count(self) -> int:
        if self._fitz_doc is not None:
            return self._fitz_doc.page_count
        if self._pdfium_doc is not None:
            return len(self._pdfium_doc)
        from pypdf import PdfReader

        return len(PdfReader(str(self.path)).pages)

    def render(self, index: int):
        from PIL import Image

        if self._fitz_doc is not None:
            import fitz

            page = self._fitz_doc[index]
            zoom = config.OCR_DPI / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        if self._pdfium_doc is not None:
            page = self._pdfium_doc[index]
            scale = config.OCR_DPI / 72
            bitmap = page.render(scale=scale)
            img = bitmap.to_pil()
            return img.convert("RGB")
        return None

    def embedded_images(self, index: int) -> list:
        """pypdf 提取页内图片（渲染不可用时的兜底）。"""
        from PIL import Image

        images = []
        try:
            from pypdf import PdfReader

            page = PdfReader(str(self.path)).pages[index]
            for image_file in page.images:
                try:
                    img = Image.open(io.BytesIO(image_file.data))
                    img.load()
                    if img.width >= 120 and img.height >= 120:
                        images.append(img)
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            pass
        return images

    def close(self) -> None:
        if self._fitz_doc is not None:
            self._fitz_doc.close()
        if self._pdfium_doc is not None:
            try:
                self._pdfium_doc.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------- OCR ----------------
class OcrEngine:
    """Tesseract OCR 封装，失败时记录原因但不抛出（保证入库流程不中断）。"""

    def __init__(self) -> None:
        self.last_error = ""
        self._checked = False
        self.available = False

    def _check(self) -> None:
        if self._checked:
            return
        self._checked = True
        try:
            import pytesseract

            if config.TESSERACT_CMD:
                pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
            pytesseract.get_tesseract_version()
            self.available = True
        except Exception as exc:  # noqa: BLE001
            self.available = False
            self.last_error = (
                f"Tesseract 不可用（{type(exc).__name__}: {exc}）。请安装 Tesseract-OCR 并配置 "
                "TESSERACT_CMD，同时把 chi_sim.traineddata 放入 tessdata 目录"
            )

    def image_to_text(self, image) -> str:
        self._check()
        if not self.available or image is None:
            return ""
        try:
            import pytesseract

            return pytesseract.image_to_string(image, lang=config.OCR_LANG)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"OCR 执行失败: {exc}"
            return ""


ocr_engine = OcrEngine()


def ocr_status() -> dict:
    ocr_engine._check()
    return {"available": ocr_engine.available, "message": ocr_engine.last_error,
            "lang": config.OCR_LANG, "tesseract_cmd": config.TESSERACT_CMD or "(PATH)"}


# ---------------- PDF ----------------
def parse_pdf(path: Path) -> ParseResult:
    from pypdf import PdfReader

    result = ParseResult()
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:  # noqa: BLE001
                return ParseResult(status="error", message="PDF 已加密，无法解析")
        pages = reader.pages
    except Exception as exc:  # noqa: BLE001
        return ParseResult(status="error", message=f"PDF 打开失败: {exc}")

    result.page_count = len(pages)
    renderer = PdfRenderer(path)
    texts: list[str] = []
    try:
        for index, page in enumerate(pages):
            try:
                page_text = clean_text(page.extract_text() or "")
            except Exception as exc:  # noqa: BLE001
                page_text = ""
                result.warnings.append(f"第 {index + 1} 页文本抽取失败: {exc}")

            if len(page_text) >= config.OCR_MIN_TEXT_CHARS:
                texts.append(page_text)
                continue

            # 文本层为空/过少：判定为扫描页，走 OCR
            image = renderer.render(index)
            ocr_text = clean_text(ocr_engine.image_to_text(image)) if image is not None else ""
            if not ocr_text:
                for img in renderer.embedded_images(index):
                    ocr_text += "\n" + ocr_engine.image_to_text(img)
                ocr_text = clean_text(ocr_text)
            if ocr_text:
                result.ocr_pages += 1
                texts.append(ocr_text)
            elif page_text:
                texts.append(page_text)
    finally:
        renderer.close()

    result.text = clean_text("\n\n".join(texts))
    result.source_type = "pdf-ocr" if result.ocr_pages else "pdf-text"
    if not result.text:
        result.status = "empty"
        if not ocr_engine.available:
            result.message = "未提取到文本（疑似扫描件），且 " + (ocr_engine.last_error or "OCR 不可用")
        else:
            result.message = "未提取到任何文本内容"
    return result


# ---------------- 图片 ----------------
def parse_image(path: Path) -> ParseResult:
    from PIL import Image

    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:  # noqa: BLE001
        return ParseResult(status="error", message=f"图片损坏或格式不支持: {exc}")

    text = clean_text(ocr_engine.image_to_text(img))
    result = ParseResult(text=text, page_count=1, ocr_pages=1 if text else 0, source_type="image-ocr")
    if not text:
        result.status = "empty"
        result.message = ocr_engine.last_error or "图片中未识别到文本"
    return result


# ---------------- 纯文本 ----------------
def parse_text(path: Path) -> ParseResult:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return ParseResult(status="error", message=f"文本读取失败: {exc}")
    text = clean_text(text)
    result = ParseResult(text=text, page_count=1, source_type="text")
    if not text:
        result.status = "empty"
        result.message = "文件内容为空"
    return result


def parse_file(path: Path) -> ParseResult:
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return parse_pdf(path)
    if suffix in IMAGE_SUFFIXES:
        return parse_image(path)
    if suffix in TEXT_SUFFIXES:
        return parse_text(path)
    return ParseResult(status="error", message=f"不支持的文件类型: {suffix or '未知'}")


# ---------------- 分块 ----------------
_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")


def split_chunks(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """按句切分后再按长度聚合，保留上下文重叠。"""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    text = clean_text(text)
    if not text:
        return []

    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    buffer = ""
    for sent in sentences:
        if len(buffer) + len(sent) <= size:
            buffer += sent
            continue
        if buffer:
            chunks.append(buffer)
        tail = buffer[-overlap:] if overlap and len(buffer) > overlap else ""
        buffer = tail + sent
        while len(buffer) > size:            # 超长单句硬切
            step = size - overlap if size > overlap else size
            chunks.append(buffer[:size])
            buffer = buffer[step:]
    if buffer.strip():
        chunks.append(buffer.strip())

    return [c.strip() for c in chunks if len(c.strip()) >= 2]
