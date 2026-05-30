"""将 IMPROVEMENT_REPORT.md 转换为 PDF（reportlab + 中文字体支持）"""
import os, re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── 字体注册（Windows 内置中文字体）────────────────────────────────────────
FONT_PATH = r"C:\Windows\Fonts"
fonts_to_try = [
    ("SimSun",   "simsun.ttc"),
    ("Microsoft YaHei", "msyh.ttc"),
    ("SimHei",   "simhei.ttf"),
]
MAIN_FONT = None
for fname, ffile in fonts_to_try:
    fpath = os.path.join(FONT_PATH, ffile)
    if os.path.exists(fpath):
        try:
            pdfmetrics.registerFont(TTFont(fname, fpath))
            MAIN_FONT = fname
            print(f"使用字体: {fname}")
            break
        except Exception as e:
            print(f"  {fname} 加载失败: {e}")

if not MAIN_FONT:
    raise RuntimeError("未找到可用中文字体，请安装 SimSun/Microsoft YaHei")

# ── 样式定义 ─────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm

def make_styles():
    base = getSampleStyleSheet()
    def s(name, **kw):
        kw.setdefault("fontName", MAIN_FONT)
        kw.setdefault("leading", kw.get("fontSize", 10) * 1.5)
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    return {
        "title":    s("title",    fontSize=20, textColor=colors.HexColor("#1a1a2e"),
                      spaceAfter=6, alignment=1, fontName=MAIN_FONT),
        "subtitle": s("subtitle", fontSize=11, textColor=colors.HexColor("#555"),
                      spaceAfter=4, alignment=1),
        "h1":       s("h1",       fontSize=14, textColor=colors.HexColor("#1a3a5c"),
                      spaceBefore=14, spaceAfter=4, fontName=MAIN_FONT),
        "h2":       s("h2",       fontSize=12, textColor=colors.HexColor("#2c5f8a"),
                      spaceBefore=10, spaceAfter=3, fontName=MAIN_FONT),
        "h3":       s("h3",       fontSize=10.5, textColor=colors.HexColor("#3a7abd"),
                      spaceBefore=7, spaceAfter=2, fontName=MAIN_FONT),
        "body":     s("body",     fontSize=9.5, spaceAfter=3, leading=15),
        "bullet":   s("bullet",   fontSize=9.5, spaceAfter=2, leading=15,
                      leftIndent=14, firstLineIndent=-10),
        "code":     s("code",     fontSize=8.5, fontName="Courier",
                      textColor=colors.HexColor("#2d3748"), leading=13,
                      backColor=colors.HexColor("#f7f7f7"),
                      leftIndent=12, spaceAfter=4),
        "note":     s("note",     fontSize=8.5, textColor=colors.HexColor("#666"),
                      spaceAfter=2, leading=13, alignment=1),
    }

# ── Markdown 解析（简化版，不依赖 markdown 库）───────────────────────────
def parse_md(text, styles):
    story = []
    lines = text.split("\n")
    i = 0
    in_table = False
    table_rows = []
    in_code = False
    code_buf = []

    def flush_table():
        nonlocal in_table, table_rows
        if not table_rows:
            return
        col_n = max(len(r) for r in table_rows)
        # 过滤分隔行
        data = [r for r in table_rows if not all(set(c.strip()) <= set("-|: ") for c in r)]
        if not data:
            table_rows = []; in_table = False; return
        # 等宽处理
        data = [r + [""] * (col_n - len(r)) for r in data]
        avail_w = PAGE_W - 2 * MARGIN
        col_w = avail_w / col_n
        ts = TableStyle([
            ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#1a3a5c")),
            ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
            ("FONTNAME",    (0,0), (-1,-1), MAIN_FONT),
            ("FONTSIZE",    (0,0), (-1,-1), 8.5),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#cdd5e0")),
            ("TOPPADDING",  (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1), 4),
            ("LEFTPADDING", (0,0), (-1,-1), 6),
            ("ALIGN",       (0,0), (-1,-1), "LEFT"),
            ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ])
        def _esc(s):
            s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
            s = re.sub(r"`(.+?)`", r'<font name="Courier" size="8.5">\1</font>', s)
            return s
        tbl_data = [[Paragraph(_esc(cell), styles["body"]) for cell in row] for row in data]
        tbl = Table(tbl_data, colWidths=[col_w]*col_n, repeatRows=1)
        tbl.setStyle(ts)
        story.append(tbl)
        story.append(Spacer(1, 6))
        table_rows = []; in_table = False

    while i < len(lines):
        line = lines[i]

        # 代码块
        if line.strip().startswith("```"):
            if in_code:
                if code_buf:
                    story.append(Paragraph("<br/>".join(
                        ln.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                        for ln in code_buf), styles["code"]))
                code_buf = []; in_code = False
            else:
                in_code = True
            i += 1; continue
        if in_code:
            code_buf.append(line); i += 1; continue

        # 表格
        if "|" in line and line.strip().startswith("|"):
            flush_table() if not in_table else None
            in_table = True
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            table_rows.append(cells)
            i += 1; continue
        elif in_table:
            flush_table()

        stripped = line.strip()

        if not stripped:
            story.append(Spacer(1, 4)); i += 1; continue

        # 标题
        if stripped.startswith("# ") and not stripped.startswith("## "):
            flush_table()
            txt = stripped[2:].strip()
            story.append(Paragraph(txt, styles["title"]))
            story.append(HRFlowable(width="100%", thickness=1.5,
                                    color=colors.HexColor("#1a3a5c"), spaceAfter=8))
            i += 1; continue
        if stripped.startswith("## "):
            flush_table()
            txt = stripped[3:].strip()
            story.append(Paragraph(txt, styles["h1"]))
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.HexColor("#2c5f8a"), spaceAfter=4))
            i += 1; continue
        if stripped.startswith("### "):
            flush_table()
            txt = stripped[4:].strip()
            story.append(Paragraph(txt, styles["h2"]))
            i += 1; continue
        if stripped.startswith("#### "):
            flush_table()
            txt = stripped[5:].strip()
            story.append(Paragraph(txt, styles["h3"]))
            i += 1; continue

        # 水平线
        if set(stripped) <= set("-"):
            story.append(HRFlowable(width="100%", thickness=0.3,
                                    color=colors.HexColor("#ccc"), spaceAfter=4))
            i += 1; continue

        # 强调和内联格式转换
        def inline(t):
            t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
            t = re.sub(r"`(.+?)`", r'<font name="Courier" size="8.5">\1</font>', t)
            t = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", t)
            return t

        # 列表
        if stripped.startswith("- ") or stripped.startswith("* "):
            flush_table()
            txt = inline(stripped[2:])
            story.append(Paragraph(f"• {txt}", styles["bullet"]))
            i += 1; continue
        if re.match(r"^\d+\.", stripped):
            flush_table()
            txt = inline(re.sub(r"^\d+\.\s*", "", stripped))
            story.append(Paragraph(f"• {txt}", styles["bullet"]))
            i += 1; continue

        # 注释行（斜体/引用）
        if stripped.startswith("> ") or stripped.startswith("*注"):
            txt = inline(stripped.lstrip("> "))
            story.append(Paragraph(txt, styles["note"]))
            i += 1; continue

        # 普通段落
        flush_table()
        story.append(Paragraph(inline(stripped), styles["body"]))
        i += 1

    flush_table()
    return story

# ── 主程序 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    md_path  = os.path.join(here, "IMPROVEMENT_REPORT.md")
    pdf_path = os.path.join(here, "OilSense_改进报告.pdf")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=2*cm, bottomMargin=2*cm,
        title="OilSense 系统改进报告",
        author="Heimdallrs",
    )

    styles = make_styles()
    story  = parse_md(md_text, styles)

    doc.build(story)
    print(f"PDF 已生成：{pdf_path}")
    print(f"文件大小：{os.path.getsize(pdf_path)//1024} KB")
