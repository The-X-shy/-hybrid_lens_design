from __future__ import annotations

import math
import shutil
import tempfile
import zipfile
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/Users/lilin/Desktop/hybrid_lens_design')
DOC_DIR = ROOT / '文档' / '汇报PPT'
ASSET_DIR = DOC_DIR / 'progress_report_20260306_assets'
OUT_PPTX = DOC_DIR / '毕业设计进度汇报_20260306.pptx'
DESIGN_MD = DOC_DIR / 'design_philosophy_progress_report.md'
TEMPLATE_PPTX = ROOT / '文档' / '中期报告' / 'midterm_report_template_style.pptx'
FIG_DIR = ROOT / '文档' / '毕业论文' / 'latex_strict_bit' / 'figures'
TECH_ROUTE_PDF = ROOT / '文档' / '毕业论文' / 'latex_strict_bit' / '技术路线图.pdf'

W, H = 1920, 1080
EMU_W, EMU_H = 12192000, 6858000

BG = '#F7F3EC'
CARD = '#FFFDFC'
NAVY = '#14384D'
NAVY_2 = '#1D4E66'
ORANGE = '#C76D3C'
SAND = '#EADCCE'
LINE = '#D7DFE6'
MUTED = '#54616B'
GREEN = '#6A8B6F'
RED = '#B8574A'
LIGHT_BLUE = '#E6EEF4'
FOOT = '#0F3351'
WHITE = '#FFFFFF'

FONT_TITLE = '/System/Library/Fonts/Hiragino Sans GB.ttc'
FONT_BOLD = '/System/Library/Fonts/STHeiti Medium.ttc'
FONT_BODY = '/System/Library/Fonts/Hiragino Sans GB.ttc'
FONT_SERIF = '/System/Library/Fonts/Supplemental/Times New Roman.ttf'


def ensure_dirs() -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def title_font(size: int) -> ImageFont.FreeTypeFont:
    return font(FONT_BOLD, size)


def body_font(size: int) -> ImageFont.FreeTypeFont:
    return font(FONT_BODY, size)


def serif_font(size: int) -> ImageFont.FreeTypeFont:
    return font(FONT_SERIF, size)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for raw in text.split('\n'):
        if not raw:
            lines.append('')
            continue
        cur = ''
        for ch in raw:
            test = cur + ch
            if draw.textlength(test, font=fnt) <= max_width or not cur:
                cur = test
            else:
                lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
    return lines


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], fnt: ImageFont.FreeTypeFont,
                 fill: str = NAVY, line_gap: int = 10, align: str = 'left') -> int:
    x, y, w, h = box
    lines = wrap_text(draw, text, fnt, w)
    line_h = fnt.size + line_gap
    total_h = len(lines) * line_h - line_gap
    y0 = y
    for line in lines:
        tw = draw.textlength(line, font=fnt)
        tx = x if align == 'left' else x + (w - tw) / 2
        draw.text((tx, y0), line, font=fnt, fill=fill)
        y0 += line_h
    return y0 - y + (0 if not lines else -line_gap)


def draw_bullets(draw: ImageDraw.ImageDraw, items: list[str], box: tuple[int, int, int, int],
                 fnt: ImageFont.FreeTypeFont, fill: str = NAVY, bullet_color: str = ORANGE,
                 gap: int = 18) -> None:
    x, y, w, h = box
    cy = y
    for item in items:
        draw.ellipse((x, cy + 11, x + 10, cy + 21), fill=bullet_color)
        used = draw_wrapped(draw, item, (x + 24, cy, w - 24, h), fnt, fill=fill, line_gap=8)
        cy += used + gap


def card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str = CARD,
         outline: str = LINE, radius: int = 26, width: int = 2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def badge(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str, txt: str = WHITE) -> None:
    fnt = body_font(22)
    tw = draw.textlength(text, font=fnt)
    box = (xy[0], xy[1], int(xy[0] + tw + 34), xy[1] + 42)
    draw.rounded_rectangle(box, radius=20, fill=fill)
    draw.text((xy[0] + 17, xy[1] + 8), text, font=fnt, fill=txt)


def paste_contain(base: Image.Image, img: Image.Image, box: tuple[int, int, int, int], bg: str = CARD) -> None:
    x, y, w, h = box
    img = img.convert('RGB')
    ratio = min(w / img.width, h / img.height)
    nw, nh = max(1, int(img.width * ratio)), max(1, int(img.height * ratio))
    resized = img.resize((nw, nh), Image.LANCZOS)
    px = x + (w - nw) // 2
    py = y + (h - nh) // 2
    base.paste(resized, (px, py))


def slide_canvas(title: str, footer: str, page_no: int, kicker: str | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, W, 92), fill=LIGHT_BLUE)
    draw.rectangle((0, H - 42, W, H), fill=FOOT)
    draw.text((80, 120), title, font=title_font(46), fill=NAVY)
    if kicker:
        badge(draw, (80, 54), kicker, ORANGE)
    draw.text((W - 140, H - 34), f'{page_no:02d}', font=serif_font(20), fill=WHITE)
    draw.text((80, H - 34), footer, font=body_font(20), fill=WHITE)
    return img, draw


def render_pdf_first_page(pdf_path: Path, out_png: Path) -> None:
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2.6, 2.6), alpha=False)
    pix.save(out_png)


def load_img(name: str) -> Image.Image:
    return Image.open(FIG_DIR / name)


def make_cover(route_png: Path) -> Image.Image:
    img, draw = slide_canvas('毕业设计进度汇报', '端到端折衍混合成像系统设计研究', 1, 'PROGRESS REPORT')
    draw_wrapped(draw, '端到端折衍混合成像系统设计研究', (80, 210, 820, 100), title_font(64), NAVY)
    draw_wrapped(draw, '当前阶段：主线实验已跑通，报告与论文进入集中修订阶段', (82, 330, 780, 70), body_font(28), MUTED)
    card(draw, (78, 430, 838, 730), fill=CARD)
    items = [
        '学院：光电学院    专业：测控技术与仪器',
        '学生：李俊贤    学号：1120221991',
        '指导教师：张楠    汇报时间：2026年3月',
    ]
    draw_bullets(draw, items, (118, 480, 660, 220), body_font(26), fill=NAVY_2, bullet_color=ORANGE)
    card(draw, (980, 180, 1800, 900), fill='#FCF8F2', outline='#E0D3C5')
    route_img = Image.open(route_png)
    paste_contain(img, route_img, (1030, 230, 720, 620))
    draw.text((1050, 850), '当前汇报重点：进度、结果、文档节点', font=body_font(24), fill=MUTED)
    return img


def make_agenda() -> Image.Image:
    img, draw = slide_canvas('汇报提纲', '明日汇报版 · 内容与论文当前口径一致', 2, 'AGENDA')
    items = [
        ('01', '研究目标与问题定义'),
        ('02', '技术路线与当前进展'),
        ('03', '关键实验结果与主要结论'),
        ('04', '论文与中期报告修改进度'),
        ('05', '后续计划与时间节点'),
        ('06', '需要老师确认的问题'),
    ]
    for idx, (num, title) in enumerate(items):
        x = 100 if idx < 3 else 980
        y = 220 + (idx % 3) * 210
        card(draw, (x, y, x + 760, y + 150), fill=CARD)
        draw.text((x + 36, y + 34), num, font=serif_font(48), fill=ORANGE)
        draw_wrapped(draw, title, (x + 130, y + 44, 560, 60), title_font(30), NAVY)
    return img


def make_goals() -> Image.Image:
    img, draw = slide_canvas('研究目标与问题定义', '问题定义保持与毕业论文第1章一致', 3, 'GOALS')
    left = (80, 190, 900, 950)
    card(draw, left)
    draw.text((110, 220), '核心目标', font=title_font(34), fill=NAVY)
    draw_bullets(draw, [
        '构建折射传播与 DOE 波动传播统一的可微分建模链路。',
        '采用“折射基线 → 折衍联合 → 端到端协同”的分阶段优化流程。',
        '建立统一复评与环境锁定机制，保证实验结果可追溯、可复核。',
    ], (110, 285, 720, 430), body_font(27), fill=NAVY_2)
    xs = [980, 980, 980]
    ys = [190, 390, 590]
    titles = ['问题一：建模耦合', '问题二：训练稳定性', '问题三：评估一致性']
    texts = [
        '射线追迹与波动传播物理模型不同，需在同一计算图中衔接。',
        '折射、DOE 与网络参数尺度差异大，直接联合优化易震荡。',
        '镜头快照、PSF 缓存与评估脚本必须统一，否则指标会失真。',
    ]
    colors = [LIGHT_BLUE, '#F8EBDD', '#E6F0E7']
    for x, y, t, body, fc in zip(xs, ys, titles, texts, colors):
        card(draw, (x, y, 1820, y + 150), fill=fc)
        draw_wrapped(draw, t, (1010, y + 24, 760, 40), title_font(28), NAVY)
        draw_wrapped(draw, body, (1010, y + 72, 760, 54), body_font(24), MUTED)
    return img


def make_route(route_png: Path) -> Image.Image:
    img, draw = slide_canvas('技术路线', '当前技术路线已联动更新到论文第1章框架图', 4, 'ROUTE')
    draw_bullets(draw, [
        '主线顺序：折射基线优化 → 折衍联合优化 → 端到端协同训练。',
        '当前已补充 DOE 解冻实验，并将统一复评纳入主证据链。',
        '现阶段结论可直接支撑中期汇报与毕业论文主表。',
    ], (90, 190, 780, 220), body_font(27), fill=NAVY_2)
    card(draw, (80, 410, 1840, 950), fill='#FCF8F2', outline='#E2D5C7')
    route_img = Image.open(route_png)
    paste_contain(img, route_img, (120, 450, 1680, 460))
    return img


def make_progress_overview() -> Image.Image:
    img, draw = slide_canvas('当前进展总览', '以“实验闭环 + 文档修订”作为当前汇报主线', 5, 'STATUS')
    boxes = [
        (80, 220, 560, 650, '已完成', GREEN, [
            '项目本地重构与环境固定',
            '统一日志与复评流程',
            '6组主实验与 DOE 解冻补充实验',
            '论文主结论与图表口径统一',
        ]),
        (680, 220, 560, 650, '进行中', ORANGE, [
            '中期报告回改与汇报材料整理',
            '毕业论文版式压缩与图表重排',
            '参考文献与正文细节继续核查',
        ]),
        (1280, 220, 560, 650, '待确认/待收口', RED, [
            '是否继续补做多种子或制造约束实验',
            '与导师确认最终补充实验边界',
            '按意见完成终稿收口',
        ]),
    ]
    for x, y, w, h, title, color, items in boxes:
        card(draw, (x, y, x + w, y + h), fill=CARD)
        badge(draw, (x + 30, y + 24), title, color)
        draw_bullets(draw, items, (x + 34, y + 100, w - 68, h - 140), body_font(24), fill=NAVY_2, bullet_color=color)
    return img


def make_stage_results() -> Image.Image:
    img, draw = slide_canvas('阶段结果：折射与折衍优化', '当前光学前端主线结果已经稳定', 6, 'RESULTS I')
    draw_bullets(draw, [
        'GeoLens 阶段：RMS 弥散斑半径 0.3072 mm → 0.0967 mm，下降约 68%。',
        'Hybrid 阶段：PSF 损失 0.02106 → 0.00961，下降约 54%。',
        '结论：折射基线已可用，DOE 参数有效参与优化，不是单纯靠网络补偿。',
    ], (80, 180, 1760, 150), body_font(25), fill=NAVY_2)
    boxes = [(80, 380, 760, 500), (880, 380, 420, 500), (1340, 380, 500, 500)]
    labels = ['GeoLens 优化后布局', 'DOE 相位图', 'Hybrid PSF 损失曲线']
    files = ['geolens_layout.png', 'doe_phase_map.png', 'hybrid_loss_curve.png']
    for (x, y, w, h), label, fname in zip(boxes, labels, files):
        card(draw, (x, y, x + w, y + h), fill=CARD)
        paste_contain(img, load_img(fname), (x + 24, y + 24, w - 48, h - 90))
        draw_wrapped(draw, label, (x + 30, y + h - 54, w - 60, 36), body_font(22), MUTED, align='center')
    return img


def draw_table(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], headers: list[str], rows: list[list[str]], highlight_rows: set[int] | None = None) -> None:
    highlight_rows = highlight_rows or set()
    x, y, w, h = box
    cols = [0.11, 0.37, 0.17, 0.15, 0.20]
    col_x = [x]
    for c in cols[:-1]:
        col_x.append(col_x[-1] + int(w * c))
    header_h = 58
    row_h = int((h - header_h) / len(rows))
    draw.rounded_rectangle((x, y, x + w, y + h), radius=20, fill=CARD, outline=LINE, width=2)
    draw.rectangle((x, y, x + w, y + header_h), fill=LIGHT_BLUE)
    for xx in col_x[1:]:
        draw.line((xx, y, xx, y + h), fill=LINE, width=2)
    for i in range(len(rows) + 1):
        yy = y + header_h + i * row_h
        draw.line((x, yy, x + w, yy), fill=LINE, width=2)
    hf = title_font(20)
    bf = body_font(19)
    curx = x
    widths = [int(w * c) for c in cols]
    for text, cw in zip(headers, widths):
        tw = draw.textlength(text, font=hf)
        draw.text((curx + (cw - tw) / 2, y + 16), text, font=hf, fill=NAVY)
        curx += cw
    for r, row in enumerate(rows):
        yy = y + header_h + r * row_h
        if r in highlight_rows:
            draw.rectangle((x + 2, yy + 2, x + w - 2, yy + row_h - 2), fill='#FFF4EA')
        curx = x
        for c, (text, cw) in enumerate(zip(row, widths)):
            fill = ORANGE if r in highlight_rows and c in (0, 2) else NAVY_2
            f = body_font(18) if c != 1 else body_font(17)
            box_x = curx + 10
            if c == 1:
                draw_wrapped(draw, text, (box_x, yy + 12, cw - 20, row_h - 16), f, fill)
            else:
                tw = draw.textlength(text, font=f)
                draw.text((curx + (cw - tw) / 2, yy + 16), text, font=f, fill=fill)
            curx += cw


def make_ablation_table() -> Image.Image:
    img, draw = slide_canvas('6组实验结果对比', '当前主表口径：统一复评（initial lens）', 7, 'RESULTS II')
    headers = ['实验', '配置', 'PSNR', 'SSIM', 'LPIPS']
    rows = [
        ['Exp3', '纯折射 + Plain UNet', '31.17', '0.909', '0.028'],
        ['Exp2', '纯折射 + MHA-UNet', '29.89', '0.904', '0.031'],
        ['Exp4', '混合 + Plain UNet', '21.89', '0.578', '0.250'],
        ['Exp1', '混合 + MHA-UNet', '21.59', '0.608', '0.255'],
        ['Exp5', '混合 + Plain UNet + DOE解冻', '21.95', '0.581', '0.246'],
        ['Exp6', '混合 + MHA-UNet + DOE解冻', '22.26', '0.630', '0.232'],
    ]
    draw_table(draw, (80, 190, 1760, 560), headers, rows, highlight_rows={0, 5})
    card(draw, (80, 790, 1840, 980), fill='#FCF8F2', outline='#E2D5C7')
    draw_bullets(draw, [
        '纯折射组中，Plain UNet 仍优于 MHA-UNet。',
        '混合组在 DOE 冻结条件下呈现指标权衡，而不是单边优势。',
        'DOE 解冻后两类网络都提升，其中 Exp6 是当前 DOE 启用方案中的最佳结果。',
    ], (110, 826, 1680, 120), body_font(24), fill=NAVY_2)
    return img


def make_findings() -> Image.Image:
    img, draw = slide_canvas('主要结论与问题修复', '本轮汇报的价值在于主结论已收敛，口径已统一', 8, 'FINDINGS')
    card(draw, (80, 190, 940, 950), fill=CARD)
    draw.text((110, 224), '主要结论', font=title_font(34), fill=NAVY)
    draw_bullets(draw, [
        '分阶段流程已经打通：折射基线、DOE 优化、端到端协同均能稳定运行。',
        '混合系统在 DOE 冻结时呈网络指标权衡；DOE 解冻后性能继续提升。',
        '当前最值得保留的主结论：Exp6 是 DOE 启用方案中的最佳结果。',
    ], (110, 280, 760, 300), body_font(25), fill=NAVY_2)
    card(draw, (980, 190, 1840, 550), fill='#FCF8F2', outline='#E2D5C7')
    draw.text((1010, 224), '已修复的问题', font=title_font(34), fill=NAVY)
    draw_bullets(draw, [
        '评估脚本曾使用训练后漂移的镜头参数，导致指标失真。',
        '现已统一采用 initial lens 进行复评，主表与论文结论同步更新。',
    ], (1010, 280, 770, 180), body_font(25), fill=NAVY_2)
    card(draw, (980, 580, 1400, 950), fill=CARD)
    card(draw, (1420, 580, 1840, 950), fill=CARD)
    paste_contain(img, load_img('eval_exp3_example.png'), (1005, 608, 370, 265))
    paste_contain(img, load_img('eval_exp1_example.png'), (1445, 608, 370, 265))
    draw_wrapped(draw, '纯折射 + Plain UNet', (1010, 885, 360, 28), body_font(22), MUTED, align='center')
    draw_wrapped(draw, '混合 + MHA-UNet', (1450, 885, 360, 28), body_font(22), MUTED, align='center')
    return img


def make_doc_progress() -> Image.Image:
    img, draw = slide_canvas('文档与归档进度', '实验、论文、过程记录目前已经基本同口径', 9, 'DOCUMENTS')
    cards = [
        ('中期报告', 'LaTeX 版本已成形\n当前重点是按老师意见回改和压缩表达。', '进行中', ORANGE),
        ('毕业论文', '当前约 57 页，可稳定编译。\n实验章、方法章、结论与技术路线图已联动更新。', '进行中', ORANGE),
        ('项目归档', 'results / figures / logs / 文档目录已分离整理。\n主结果、复评结果与过程记录均可回溯。', '已完成', GREEN),
    ]
    for i, (title, body, status, color) in enumerate(cards):
        x = 100
        y = 210 + i * 230
        card(draw, (x, y, 1820, y + 180), fill=CARD)
        badge(draw, (x + 30, y + 26), status, color)
        draw_wrapped(draw, title, (x + 200, y + 34, 260, 38), title_font(30), NAVY)
        draw_wrapped(draw, body, (x + 200, y + 86, 1520, 70), body_font(24), NAVY_2)
    return img


def make_timeline() -> Image.Image:
    img, draw = slide_canvas('进度表与时间节点', '按你明天汇报需求补入的简单进度表', 10, 'TIMELINE')
    card(draw, (100, 180, 1820, 430), fill=CARD)
    headers = ['时间', '目标', '状态']
    rows = [
        ['当前（2026-03-06）', '主实验与统一复评已完成；正在集中修改中期报告与毕业论文', '进行中'],
        ['2026-03-31 前', '改完中期报告', '计划完成'],
        ['2026-04-30 前', '改完毕业论文', '计划完成'],
    ]
    # simple 3-col table
    x, y, w, h = 130, 220, 1660, 330
    widths = [340, 970, 350]
    header_h = 60
    row_h = 90
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill='#FFF9F2', outline='#E0D3C5', width=2)
    draw.rectangle((x, y, x + w, y + header_h), fill=LIGHT_BLUE)
    vx = x
    for ww in widths[:-1]:
        vx += ww
        draw.line((vx, y, vx, y + h), fill='#E0D3C5', width=2)
    for i in range(3):
        yy = y + header_h + i * row_h
        draw.line((x, yy, x + w, yy), fill='#E0D3C5', width=2)
    curx = x
    for head, ww in zip(headers, widths):
        tw = draw.textlength(head, font=title_font(22))
        draw.text((curx + (ww - tw) / 2, y + 17), head, font=title_font(22), fill=NAVY)
        curx += ww
    for r, row in enumerate(rows):
        yy = y + header_h + r * row_h
        curx = x
        for c, (txt, ww) in enumerate(zip(row, widths)):
            if c == 2:
                badge(draw, (curx + 75, yy + 24), txt, ORANGE if '进行' in txt else GREEN)
            else:
                draw_wrapped(draw, txt, (curx + 18, yy + 18, ww - 36, row_h - 20), body_font(22), NAVY_2)
            curx += ww
    card(draw, (100, 660, 1820, 920), fill=CARD)
    draw.text((130, 700), '节点说明', font=title_font(30), fill=NAVY)
    draw_bullets(draw, [
        '三月底前把中期报告按老师意见完成一轮定稿。',
        '四月底前把毕业论文按最终实验口径完成系统修改与收口。',
        '后续汇报口径优先以“6组实验 + 统一复评 + 论文修订进度”展开。',
    ], (130, 750, 1620, 120), body_font(24), fill=NAVY_2)
    return img


def make_next_steps() -> Image.Image:
    img, draw = slide_canvas('下一步工作与风险', '明天汇报时建议把“收口”放在主叙事里', 11, 'NEXT')
    card(draw, (80, 190, 940, 950), fill=CARD)
    card(draw, (980, 190, 1840, 950), fill='#FCF8F2', outline='#E2D5C7')
    draw.text((110, 224), '下一步工作', font=title_font(34), fill=NAVY)
    draw_bullets(draw, [
        '继续根据老师意见修改中期报告。',
        '继续精修毕业论文排版、图表和文字。',
        '根据答复决定是否补做多种子、制造约束或空间变化 PSF 实验。',
    ], (110, 280, 760, 320), body_font(25), fill=NAVY_2)
    draw.text((1010, 224), '需要老师确认', font=title_font(34), fill=NAVY)
    draw_bullets(draw, [
        '毕业论文终稿是否继续扩大补充实验范围。',
        '汇报重点是突出方法闭环，还是突出最优指标。',
        '后续时间分配应更多放在实验，还是更多放在文档收口。',
    ], (1010, 280, 760, 320), body_font(25), fill=NAVY_2)
    badge(draw, (110, 870), '建议主线：实验已闭环，当前重心转向文档定稿', GREEN)
    return img


def make_closing() -> Image.Image:
    img, draw = slide_canvas('请老师指导', '阶段性结果已收敛，后续重心是文档收口与边界确认', 12, 'Q & A')
    draw_wrapped(draw, '当前阶段判断：\n主线实验已经具备支撑本科毕设答辩的基础。', (140, 250, 1640, 180), title_font(54), NAVY, line_gap=16, align='center')
    draw_wrapped(draw, '明天汇报希望重点确认：补充实验是否继续扩展，以及中期报告/毕业论文的修改边界。', (220, 510, 1480, 80), body_font(30), MUTED, align='center')
    card(draw, (420, 690, 1500, 830), fill='#FCF8F2', outline='#E2D5C7')
    draw_wrapped(draw, '谢谢老师。', (420, 730, 1080, 40), title_font(36), NAVY, align='center')
    return img


def save_slide(img: Image.Image, index: int) -> Path:
    out = ASSET_DIR / f'slide_{index:02d}.png'
    img.save(out)
    return out


def write_design_md() -> None:
    DESIGN_MD.write_text(
        '# Quiet Engineering\n\n'
        '这套进度汇报采用“安静工程化”的视觉方向：浅暖底色、深蓝主色、橙色节点强调，避免炫技式装饰，把重点放在结构、证据和节奏上。画面重心优先服务于三类信息：研究主线、阶段结果、时间节点。\n\n'
        '空间组织上使用大量留白和卡片化分区，使“已完成 / 进行中 / 下一步”在视觉上自然分层。图像只承担证据展示角色，不与文字抢主导。所有高频信息，如实验结论、截止日期、状态标签，都通过统一颜色和位置体系强化识别。\n\n'
        '配色上以深蓝表示方法与结论，以橙色表示进度与提醒，以浅灰蓝维持学术汇报的稳定感。这样的视觉系统更接近论文答辩或阶段汇报，而不是宣传海报，适合导师快速判断当前完成度与后续工作重心。\n\n'
        '最终目标不是“做得花”，而是让老师在几分钟内清楚看到：主线实验已经跑通，主结论已经收敛，下一步主要是文档收口与少量边界实验确认。\n',
        encoding='utf-8'
    )


def make_slide_xml(image_name: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      <p:pic>
        <p:nvPicPr>
          <p:cNvPr id="2" name="SlideImage"/>
          <p:cNvPicPr/>
          <p:nvPr/>
        </p:nvPicPr>
        <p:blipFill>
          <a:blip r:embed="rId2"/>
          <a:stretch><a:fillRect/></a:stretch>
        </p:blipFill>
        <p:spPr>
          <a:xfrm>
            <a:off x="0" y="0"/>
            <a:ext cx="{EMU_W}" cy="{EMU_H}"/>
          </a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
      </p:pic>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
'''


def make_slide_rels(image_name: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{image_name}"/>
</Relationships>
'''


def build_pptx(slide_pngs: list[Path]) -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with zipfile.ZipFile(TEMPLATE_PPTX) as zf:
            zf.extractall(tmp)
        media_dir = tmp / 'ppt' / 'media'
        slide_dir = tmp / 'ppt' / 'slides'
        rel_dir = slide_dir / '_rels'
        for idx, png in enumerate(slide_pngs, 1):
            image_name = f'progress_slide_{idx:02d}.png'
            shutil.copy2(png, media_dir / image_name)
            (slide_dir / f'slide{idx}.xml').write_text(make_slide_xml(image_name), encoding='utf-8')
            (rel_dir / f'slide{idx}.xml.rels').write_text(make_slide_rels(image_name), encoding='utf-8')
        with zipfile.ZipFile(OUT_PPTX, 'w', zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(tmp.rglob('*')):
                if path.is_file():
                    zf.write(path, path.relative_to(tmp).as_posix())


def main() -> None:
    ensure_dirs()
    route_png = ASSET_DIR / 'tech_route.png'
    render_pdf_first_page(TECH_ROUTE_PDF, route_png)
    write_design_md()

    slide_imgs = [
        make_cover(route_png),
        make_agenda(),
        make_goals(),
        make_route(route_png),
        make_progress_overview(),
        make_stage_results(),
        make_ablation_table(),
        make_findings(),
        make_doc_progress(),
        make_timeline(),
        make_next_steps(),
        make_closing(),
    ]
    slide_pngs = [save_slide(img, i + 1) for i, img in enumerate(slide_imgs)]
    build_pptx(slide_pngs)
    print(f'Wrote: {OUT_PPTX}')
    print(f'Assets: {ASSET_DIR}')
    print(f'Design: {DESIGN_MD}')


if __name__ == '__main__':
    main()
