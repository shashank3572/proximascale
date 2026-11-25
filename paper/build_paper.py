"""Build the revised ProximaScale manuscript as DOCX, PDF and Markdown.

All result tables are generated from committed/validation CSVs, so the document
cannot silently disagree with the measurements.

    python paper/build_paper.py
"""
import csv
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(__file__).resolve().parent
FIG = PAPER / "figures"
VALID = ROOT / "results" / "validation_2026-11-25"
OUT_STEM = "ProximaScale_IEEE_Revised_2026-11-25"

sys.path.insert(0, str(PAPER))
from content_part1 import TITLE, AUTHORS, ABSTRACT, KEYWORDS, BLOCKS_1  # noqa: E402
from content_part2 import BLOCKS_2  # noqa: E402


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def num(value, digits=2, sd=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:  # NaN
        return "-"
    s = f"{v:.{digits}f}"
    if sd is not None:
        try:
            d = float(sd)
        except (TypeError, ValueError):
            d = None
        if d is not None and d > 0:
            s += f" +/- {d:.{digits}f}"
    return s


# --------------------------------------------------------------------------
# Tables built from results
# --------------------------------------------------------------------------
def table_ii():
    rows = read_csv(VALID / "forecast_metrics_v4" / "v4_verify" / "table_ii.csv")
    labels = {"persistence": "Persistence (reactive)", "linear": "Linear extrapolation",
              "prophet": "Prophet alone", "univariate": "Univariate LSTM",
              "hybrid": "ProximaScale (served, 30-step)"}
    out = []
    for r in rows:
        if r["method"] == "ceiling":
            out.append(["Feasible ceiling (generator)", "-", "-",
                        num(r["lead_mean_s"], 1), "-", "-",
                        "34/2/0/0"])
            continue
        no_alarms = r["precision"] in ("", "nan") or float(r["precision"]) != float(r["precision"])
        out.append([
            labels.get(r["method"], r["method"]),
            num(r["rmse"], 3, r["rmse_sd"]),
            num(r["mae"], 3, r["mae_sd"]),
            num(r["lead_mean_s"], 1, r["lead_mean_s_sd"]),
            "-" if no_alarms else num(r["f1"], 3),
            num(r["false_alarms_per_day"], 2),
            num(r["crossings"], 1),
        ])
    return ("TABLE II. Forecast accuracy and event metrics on the test split "
            "(17,988 windows, 6.25 days, 36 spike onsets, threshold 75%). Baselines are "
            "deterministic and identical across dumps; the ProximaScale row is the served v4 dump. "
            "Prophet alone never forecasts above 75%, so its alarm-based F1 is undefined.",
            ["Method", "RMSE", "MAE", "Lead time (s)", "F1", "False alarms/day", "Crossings"],
            out)


def table_iii():
    shipped = {r["policy"]: r for r in read_csv(VALID / "replay_v4" / "v4_verify" / "summary.csv")}
    nofloor = {r["policy"]: r for r in
               read_csv(VALID / "replay_v4_nofloor" / "v4_verify" / "summary.csv")}
    names = [("P0 Reactive", "P0 Reactive (threshold on current CPU)"),
             ("P1 Univariate LSTM", "P1 Univariate LSTM forecaster"),
             ("P2 ProximaScale mean", "P2 ProximaScale forecast mean"),
             ("P4 Full", "P4 Original engine (all branches, no guard)"),
             ("P5 Full, static thresholds", "P5 P4 with static thresholds"),
             ("P6 Full, no cooldown", "P6 P4 without cooldown"),
             ("P7 P3, static thresholds", "P7 Upper bound + static thresholds"),
             ("P9 Tuned (config.yaml)", "P9 Shipped config.yaml")]
    out = []
    for key, label in names:
        r = shipped[key]
        out.append([label, num(r["ups"], 1), num(r["reversals_5min"], 1),
                    num(r["anticipated"], 1), num(r["false_ups_per_day"], 2),
                    num(r["mean_replicas"], 2), num(r["time_at_max_pct"], 1)])
    r = nofloor["P9 Tuned (config.yaml)"]
    out.append(["P9 with reactive floor disabled (controlled variant)",
                num(r["ups"], 1), num(r["reversals_5min"], 1), num(r["anticipated"], 1),
                num(r["false_ups_per_day"], 2), num(r["mean_replicas"], 2),
                num(r["time_at_max_pct"], 1)])
    return ("TABLE III. Decision-engine replay on the served model's test dump "
            "(mean over 5 seeds; 36 generated spikes)",
            ["Policy", "Scale-ups", "Reversals <=5 min", "Spikes anticipated",
             "False scale-ups/day", "Mean replicas", "Time at max (%)"],
            out)


def table_iv():
    rows = read_csv(ROOT / "results" / "unseen" / "forecast.csv")
    served = "v3 trained on both"
    datasets = [("fresh", "fresh (same generator, seed 7)"),
                ("shifted", "shifted (3-min ramps, 2x spikes)"),
                ("shifted2", "shifted2 (2.5-min ramps, 1.5x spikes)"),
                ("real_dev", "real_dev (121 idle real rows)")]
    out = []
    for key, label in datasets:
        subset = {r["model"]: r for r in rows if r["dataset"] == key}
        s = subset[served]
        out.append([label, num(s["rmse"], 2), f"{int(float(s['missed']))}/{int(float(s['onsets']))}",
                    num(subset["Persistence (reactive)"]["rmse"], 2),
                    num(subset["Univariate LSTM"]["rmse"], 2)])
    return ("TABLE IV. Out-of-distribution tests on one-week series and the real sample "
            "(RMSE in CPU points)",
            ["Dataset", "ProximaScale RMSE", "Spikes missed", "Persistence RMSE", "Univariate RMSE"],
            out)


def table_v():
    rows = read_csv(VALID / "forecast_metrics_v4" / "v4_verify" / "alpha_sweep.csv")
    out = []
    for r in rows:
        out.append([r["alpha"], "yes" if r["configured"] == "True" else "",
                    num(r["rmse"], 2), num(r["mae"], 2), num(r["bias"], 2)])
    return ("TABLE V. Counterfactual blend weight vs. the generator's true load "
            "(360 post-scaling target windows)",
            ["Weight on observed value", "Configured", "RMSE", "MAE", "Bias"], out)


def table_vi():
    rows = read_csv(VALID / "forecast_metrics_v4" / "v4_verify" / "anomaly_sweep.csv")
    out = []
    for r in rows:
        if r["detector"] != "decision-time":
            continue
        out.append([r["z"], num(r["flags_pct_of_decisions"], 2), num(r["precision"], 3),
                    num(r["recall"], 2), num(r["delay_median_s"], 0),
                    num(r["false_flags_per_day"], 1)])
    return ("TABLE VI. Rolling z-score anomaly detector at decision time "
            "(precision against the generator's spike intervals)",
            ["z threshold", "Flags (% of decisions)", "Precision", "Recall",
             "Median delay (s)", "False flags/day"], out)


def table_vii():
    rows = read_csv(ROOT / "results" / "timing" / "baseline" / "latency.csv")
    keep = {(r["method"], r["n_passes"], r["component"]) for r in rows}
    out = []
    for r in rows:
        comp = r["component"]
        if comp not in ("predict_load", "end_to_end", "shap_compute_shap"):
            continue
        if comp == "predict_load":
            passes = int(float(r["n_passes"]))
            label = f"ProximaScale end-to-end, {passes} MC pass" + ("" if passes == 1 else "es")
        elif comp == "end_to_end":
            label = "Univariate LSTM end-to-end"
        else:
            label = "SHAP attribution (scale-up only)"
        out.append([label, num(r["mean_ms"], 1), num(r["p50_ms"], 1), num(r["p95_ms"], 1),
                    num(r["max_ms"], 1), num(r["pct_of_interval"], 2)])
    return ("TABLE VII. Per-decision inference latency on the test machine "
            "(300 calls per row)",
            ["Component", "Mean (ms)", "p50 (ms)", "p95 (ms)", "Max (ms)", "% of 30 s interval"],
            out)


TABLE_BUILDERS = {
    "table_ii": table_ii, "table_iii": table_iii, "table_iv": table_iv,
    "table_v": table_v, "table_vi": table_vi, "table_vii": table_vii,
}


def resolve_blocks():
    blocks = BLOCKS_1 + BLOCKS_2
    resolved = []
    for block in blocks:
        if block[0] in TABLE_BUILDERS:
            resolved.append(("table",) + TABLE_BUILDERS[block[0]]())
        else:
            resolved.append(block)
    return resolved


# --------------------------------------------------------------------------
# Inline markup
# --------------------------------------------------------------------------
TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*)")


def runs(text):
    """Split into [(text, bold, italic)]."""
    out = []
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            out.append((part[2:-2], True, False))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append((part[1:-1], False, True))
        else:
            out.append((part, False, False))
    return out


def pdf_markup(text):
    return sanitize(escape(text)).replace("**", "").replace("*", "")


def sanitize(text):
    """Replace characters outside the WinAnsi range used by the PDF base fonts."""
    for bad, good in (("\u2014", " - "), ("\u2013", "-"), ("\u2018", "'"),
                      ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'),
                      ("\u2248", "~="), ("\u00d7", "x"), ("\u2264", "<="),
                      ("\u2265", ">=")):
        text = text.replace(bad, good)
    return text


# --------------------------------------------------------------------------
# DOCX
# --------------------------------------------------------------------------
def build_docx(blocks, path):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.8); sec.bottom_margin = Inches(0.8)
    sec.left_margin = Inches(0.9); sec.right_margin = Inches(0.9)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10)
    pf = normal.paragraph_format
    pf.space_after = Pt(4)
    pf.line_spacing = 1.06

    def para(text, size=10, bold=False, italic=False, align="justify",
             space_before=0, space_after=4):
        p = doc.add_paragraph()
        p.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
                       "center": WD_ALIGN_PARAGRAPH.CENTER,
                       "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        for text_, is_bold, is_italic in runs(text):
            run = p.add_run(text_)
            run.font.name = "Times New Roman"
            run.font.size = Pt(size)
            run.bold = bold or is_bold
            run.italic = italic or is_italic
        return p

    para(TITLE, size=16, bold=True, align="center", space_after=8)
    for a in AUTHORS:
        para(a, size=8.5, align="center", space_after=1)
    para("", size=6, space_after=2)
    para("Abstract - " + ABSTRACT, size=9, space_after=4)
    para("Keywords - " + KEYWORDS, size=9, italic=True, space_after=8)

    for block in blocks:
        kind = block[0]
        if kind == "h1":
            para(block[1], size=11, bold=True, align="left", space_before=10, space_after=2)
        elif kind == "h2":
            para(block[1], size=10, bold=True, italic=True, align="left",
                 space_before=6, space_after=2)
        elif kind == "p":
            para(block[1])
        elif kind == "bullets":
            for item in block[1]:
                p = para(item, size=9.5, space_after=2)
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.first_line_indent = Inches(-0.18)
                p.runs[0].text = "- " + p.runs[0].text
        elif kind == "eq":
            para(block[1], size=9.5, italic=True, align="center", space_before=2, space_after=2)
        elif kind == "fig":
            pic = doc.add_paragraph()
            pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
            pic.paragraph_format.space_before = Pt(6)
            pic.paragraph_format.space_after = Pt(2)
            pic.add_run().add_picture(str(FIG / block[1]), width=Inches(6.2))
            para(block[2], size=8.5, italic=True, align="center", space_after=8)
        elif kind == "table":
            _, caption, headers, rows = block
            para(caption, size=8.5, bold=True, align="left", space_before=8, space_after=2)
            table = doc.add_table(rows=1, cols=len(headers))
            table.style = "Table Grid"
            hdr = table.rows[0].cells
            for i, h in enumerate(headers):
                hdr[i].text = ""
                r = hdr[i].paragraphs[0].add_run(h)
                r.bold = True
                r.font.size = Pt(8)
                r.font.name = "Times New Roman"
            for row in rows:
                cells = table.add_row().cells
                for i, value in enumerate(row):
                    cells[i].text = ""
                    r = cells[i].paragraphs[0].add_run(str(value))
                    r.font.size = Pt(8)
                    r.font.name = "Times New Roman"
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
        elif kind == "refs":
            for ref in block[1]:
                p = para(ref, size=8.5, align="left", space_after=2)
                p.paragraph_format.left_indent = Inches(0.3)
                p.paragraph_format.first_line_indent = Inches(-0.3)

    doc.save(str(path))
    print(f"[ok] {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------
def build_pdf(blocks, path):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                    SimpleDocTemplate, Spacer, Table, TableStyle)
    from PIL import Image as PILImage

    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=1.9 * cm, rightMargin=1.9 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title=TITLE, author=", ".join(a.split(",")[0] for a in AUTHORS))

    body = ParagraphStyle("body", fontName="Times-Roman", fontSize=9.6, leading=12.6,
                          alignment=TA_JUSTIFY, spaceAfter=4)
    title_st = ParagraphStyle("title", parent=body, fontName="Times-Bold", fontSize=15.5,
                              leading=19, alignment=TA_CENTER, spaceAfter=8)
    author_st = ParagraphStyle("author", parent=body, fontSize=8.4, leading=10.4,
                               alignment=TA_CENTER, spaceAfter=2)
    abs_st = ParagraphStyle("abs", parent=body, fontSize=8.9, leading=11.4)
    h1 = ParagraphStyle("h1", parent=body, fontName="Times-Bold", fontSize=11,
                        leading=13.5, alignment=TA_LEFT, spaceBefore=11, spaceAfter=3)
    h2 = ParagraphStyle("h2", parent=body, fontName="Times-BoldItalic", fontSize=9.9,
                        leading=12.4, alignment=TA_LEFT, spaceBefore=7, spaceAfter=2)
    eq = ParagraphStyle("eq", parent=body, fontName="Times-Italic", alignment=TA_CENTER,
                        spaceBefore=3, spaceAfter=3)
    cap = ParagraphStyle("cap", parent=body, fontSize=8.3, leading=10.4,
                         alignment=TA_CENTER, spaceAfter=9)
    tabcap = ParagraphStyle("tabcap", parent=body, fontName="Times-Bold", fontSize=8.4,
                            leading=10.4, alignment=TA_LEFT, spaceBefore=8, spaceAfter=3)
    ref = ParagraphStyle("ref", parent=body, fontSize=8.3, leading=10.3,
                         leftIndent=12, firstLineIndent=-12, spaceAfter=2)

    story = [Paragraph(sanitize(escape(TITLE)), title_st)]
    for a in AUTHORS:
        story.append(Paragraph(sanitize(escape(a)), author_st))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b><i>Abstract</i></b> - " + sanitize(escape(ABSTRACT)), abs_st))
    story.append(Spacer(1, 3))
    story.append(Paragraph("<i>Keywords</i> - " + sanitize(escape(KEYWORDS)), abs_st))
    story.append(Spacer(1, 6))

    max_w = doc.width
    for block in blocks:
        kind = block[0]
        if kind == "h1":
            story.append(Paragraph(sanitize(escape(block[1])), h1))
        elif kind == "h2":
            story.append(Paragraph(sanitize(escape(block[1])), h2))
        elif kind == "p":
            story.append(Paragraph(pdf_markup(block[1]), body))
        elif kind == "bullets":
            for item in block[1]:
                story.append(Paragraph("&bull; " + pdf_markup(item),
                                       ParagraphStyle("b", parent=body, leftIndent=14,
                                                      firstLineIndent=-9, spaceAfter=2)))
        elif kind == "eq":
            story.append(Paragraph(sanitize(escape(block[1])), eq))
        elif kind == "fig":
            img_path = FIG / block[1]
            with PILImage.open(img_path) as im:
                w, h = im.size
            width = max_w
            height = width * h / w
            if height > 11.5 * cm:
                height = 11.5 * cm
                width = height * w / h
            story.append(KeepTogether([Spacer(1, 4), Image(str(img_path), width=width, height=height),
                                       Spacer(1, 3), Paragraph(sanitize(escape(block[2])), cap)]))
        elif kind == "table":
            _, caption, headers, rows = block
            data = [headers] + [[str(c) for c in row] for row in rows]
            font = 7.2 if len(headers) > 5 else 7.8
            table = Table(data, repeatRows=1, hAlign="CENTER")
            table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), font),
                ("LEADING", (0, 0), (-1, -1), font + 1.6),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(KeepTogether([Paragraph(sanitize(escape(caption)), tabcap), table,
                                       Spacer(1, 8)]))
        elif kind == "refs":
            for r in block[1]:
                story.append(Paragraph(sanitize(escape(r)), ref))

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Times-Roman", 8)
        canvas.drawCentredString(A4[0] / 2, 1.1 * cm, f"{doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"[ok] {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------
# Markdown (diffable source)
# --------------------------------------------------------------------------
def build_markdown(blocks, path):
    lines = [f"# {TITLE}", ""]
    for a in AUTHORS:
        lines.append(f"*{a}*  ")
    lines += ["", "**Abstract** - " + ABSTRACT, "",
              "*Keywords* - " + KEYWORDS, ""]
    for block in blocks:
        kind = block[0]
        if kind == "h1":
            lines += [f"## {block[1]}", ""]
        elif kind == "h2":
            lines += [f"### {block[1]}", ""]
        elif kind == "p":
            lines += [block[1], ""]
        elif kind == "bullets":
            lines += [f"- {i}" for i in block[1]] + [""]
        elif kind == "eq":
            lines += [f"> {block[1]}", ""]
        elif kind == "fig":
            lines += [f"![{block[2]}](figures/{block[1]})", "", f"*{block[2]}*", ""]
        elif kind == "table":
            _, caption, headers, rows = block
            lines += [f"**{caption}**", "",
                      "| " + " | ".join(headers) + " |",
                      "|" + "|".join(["---"] * len(headers)) + "|"]
            lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
            lines.append("")
        elif kind == "refs":
            lines += [f"{r}" for r in block[1]] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")
    words = len(re.sub(r"[#*|>\-]", " ", "\n".join(lines)).split())
    print(f"[ok] {path.relative_to(ROOT)} (~{words} words)")


if __name__ == "__main__":
    blocks = resolve_blocks()
    build_docx(blocks, PAPER / f"{OUT_STEM}.docx")
    build_pdf(blocks, PAPER / f"{OUT_STEM}.pdf")
    build_markdown(blocks, PAPER / f"{OUT_STEM}.md")
