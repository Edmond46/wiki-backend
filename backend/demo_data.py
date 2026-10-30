"""生成演示资料：文本型 PDF、扫描版（纯图片）PDF、图片资料，用于验证 OCR 与检索。

运行：python demo_data.py
"""
from __future__ import annotations

from pathlib import Path

# 中文字体候选（Windows 自带）
_FONT_CANDIDATES = [
    ("C:/Windows/Fonts/msyh.ttc", 0),
    ("C:/Windows/Fonts/simhei.ttf", 0),
    ("C:/Windows/Fonts/simsun.ttc", 0),
    ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0),
]

SAMPLES = [
    {
        "file": "数据库系统原理-第03讲-事务与并发控制.pdf",
        "kind": "pdf-text",
        "course": "数据库系统原理",
        "title": "第03讲 事务与并发控制",
        "pages": [
            "第03讲 事务与并发控制\n"
            "事务的 ACID 特性：原子性（Atomicity）、一致性（Consistency）、隔离性（Isolation）、持久性（Durability）。\n"
            "并发异常包括脏读、不可重复读和幻读。InnoDB 通过 MVCC 与行级锁解决并发问题。\n"
            "隔离级别分为读未提交、读已提交、可重复读和串行化，MySQL 默认使用可重复读（Repeatable Read）。\n"
            "死锁检测由 InnoDB 自动完成，出现死锁时会回滚代价较小的事务。",
            "两阶段封锁协议（2PL）：增长阶段与收缩阶段。\n"
            "redo log 保证持久性，undo log 支持回滚与 MVCC 版本链。\n"
            "日志刷盘策略 innodb_flush_log_at_trx_commit 影响数据安全与性能。\n"
            "长事务会导致 undo 日志膨胀，生产环境应避免大事务。",
        ],
    },
    {
        "file": "机器学习导论-2021-线性回归与梯度下降.pdf",
        "kind": "pdf-text",
        "course": "机器学习导论",
        "title": "2021 线性回归与梯度下降",
        "pages": [
            "2021 版讲义：线性回归与梯度下降\n"
            "线性回归假设输出是输入的线性组合，通过最小化均方误差（MSE）求解参数。\n"
            "正规方程给出闭式解，但当特征维度很高时应使用梯度下降迭代求解。\n"
            "学习率过大会导致震荡和发散，过小则收敛缓慢。",
            "随机梯度下降（SGD）每次只用一个小批量样本更新参数，计算效率高但梯度噪声大。\n"
            "动量（Momentum）与 Adam 优化器能加速收敛并缓解震荡。\n"
            "L2 正则化等价于权重衰减，可缓解过拟合；L1 正则化产生稀疏解。",
        ],
    },
    {
        "file": "操作系统-虚拟内存与页面置换.pdf",
        "kind": "pdf-text",
        "course": "操作系统",
        "title": "虚拟内存与页面置换",
        "pages": [
            "虚拟内存与页面置换\n"
            "虚拟内存让进程拥有连续的地址空间，通过页表完成虚拟地址到物理地址的映射。\n"
            "缺页中断发生时，操作系统从磁盘调入页面，必要时选择一个页面换出。\n"
            "常见页面置换算法：FIFO、LRU、Clock 与 OPT。LRU 使用最近最少使用原则，"
            "但维护成本较高，实际系统常用 Clock 及其改进算法。",
            "TLB 是页表的高速缓存，命中时无需访问内存中的页表。\n"
            "多级页表减少页表自身占用的空间。工作集模型用于防止抖动（thrashing）。",
        ],
    },
    {
        "file": "计算机网络-拥塞控制与TCP.pdf",
        "kind": "pdf-text",
        "course": "计算机网络",
        "title": "TCP 拥塞控制",
        "pages": [
            "TCP 拥塞控制\n"
            "TCP 通过慢启动、拥塞避免、快速重传与快速恢复四个阶段调节发送速率。\n"
            "拥塞窗口 cwnd 与接收窗口 rwnd 共同决定发送窗口大小。\n"
            "当发生超时重传时，ssthresh 设为当前 cwnd 的一半，并重新进入慢启动。",
            "三次握手建立连接，四次挥手释放连接。TIME_WAIT 状态持续 2MSL。\n"
            "现代内核支持 CUBIC 与 BBR 拥塞控制算法，BBR 基于带宽与时延建模。",
        ],
    },
    {
        "file": "扫描件-数据结构-考试重点-2023.jpg",
        "kind": "image",
        "course": "数据结构",
        "title": "2023 期末复习重点（扫描件）",
        "text": "数据结构期末复习重点\n"
                "一、时间复杂度分析：常见复杂度 O(1) < O(log n) < O(n) < O(n log n) < O(n^2)。\n"
                "二、二叉搜索树的中序遍历结果有序，平衡二叉树 AVL 通过旋转保持高度平衡。\n"
                "三、图的最短路径算法：Dijkstra 适用于非负权图，Floyd 可求任意两点间最短路径。\n"
                "四、排序算法：快速排序平均时间复杂度 O(n log n)，稳定性方面归并排序稳定，快排不稳定。",
    },
    {
        "file": "扫描件-计算机组成原理-总线与存储层次.pdf",
        "kind": "pdf-scan",
        "course": "计算机组成原理",
        "title": "总线与存储层次（扫描件）",
        "pages": [
            "计算机组成原理 第五章\n"
            "存储层次结构自顶向下为寄存器、Cache、主存与辅存，越靠上速度越快容量越小。\n"
            "Cache 映射方式包括直接映射、全相联映射与组相联映射，"
            "局部性原理是 Cache 能起作用的关键。",
            "总线按功能分为数据总线、地址总线与控制总线。\n"
            "总线宽度决定了单次传输的数据位数，总线带宽等于频率乘以位宽除以八。\n"
            "中断方式相比轮询方式能显著降低 CPU 占用。",
        ],
    },
]


def _font(size: int):
    from PIL import ImageFont

    for path, index in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size, index=index)
            except Exception:  # noqa: BLE001
                continue
    # 默认位图字体不支持中文，直接报错比生成乱码更好
    raise RuntimeError("未找到可用的中文字体（msyh.ttc / simhei.ttf），无法渲染演示图片")


def _render_page(text: str, width=1240, height=1754):
    """把文本渲染为 A4 比例白底黑字图片（模拟扫描件）。"""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _font(42)
    y = 120
    for line in text.split("\n"):
        # 简单按像素宽度折行
        buf = ""
        for ch in line:
            if draw.textlength(buf + ch, font=font) > width - 160:
                draw.text((80, y), buf, font=font, fill=(20, 20, 20))
                y += 70
                buf = ch
            else:
                buf += ch
        draw.text((80, y), buf, font=font, fill=(20, 20, 20))
        y += 80
    return img


def _reportlab_font() -> str:
    """注册中文字体，返回字体名（reportlab 内置字体不支持中文）。"""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for path, _ in _FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont("CJK", path))
            return "CJK"
        except Exception:  # noqa: BLE001 - ttc 集合部分格式不支持
            continue
    raise RuntimeError("未找到可用的中文字体，无法生成演示 PDF")


def _write_text_pdf(path: Path, pages: list[str], font_name: str) -> None:
    """生成带真实文本层的 PDF（模拟电子版讲义）。"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas

    width, height = A4
    font_size, leading = 12, 18
    margin, top = 20 * mm, 20 * mm
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle(path.stem)
    for page_text in pages:
        y = height - top
        for raw_line in page_text.split("\n"):
            buf = ""
            for ch in raw_line:
                if stringWidth(buf + ch, font_name, font_size) > width - 2 * margin:
                    c.setFont(font_name, font_size)
                    c.drawString(margin, y, buf)
                    y -= leading
                    buf = ch
                else:
                    buf += ch
            c.setFont(font_name, font_size)
            c.drawString(margin, y, buf)
            y -= leading
        c.showPage()
    c.save()


def _write_image_pdf(path: Path, pages: list[str]) -> None:
    """把渲染好的整页图片打包成 PDF，即纯图片（扫描版）PDF。"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    width, height = A4
    c = canvas.Canvas(str(path), pagesize=A4)
    for page_text in pages:
        img = _render_page(page_text)
        c.drawImage(ImageReader(img), 0, 0, width=width, height=height)
        c.showPage()
    c.save()


def build_samples(out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for spec in SAMPLES:
        path = out_dir / spec["file"]
        if spec["kind"] == "pdf-text":
            _write_text_pdf(path, spec["pages"], _reportlab_font())
        elif spec["kind"] == "pdf-scan":
            _write_image_pdf(path, spec["pages"])
        elif spec["kind"] == "image":
            _render_page(spec["text"]).save(path, quality=92)
        items.append({"path": str(path), "course": spec["course"], "title": spec["title"]})
    return items


if __name__ == "__main__":
    import config

    for item in build_samples(config.BASE_DIR / "data" / "samples"):
        print(item["path"])
