"""
Converts SYNOPSIS_REPORT.md into a high-quality PDF document using
Headless Chrome / Edge print-to-pdf engine.
"""

import os
import subprocess
import mistune

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(BASE_DIR, "SYNOPSIS_REPORT.md")
HTML_PATH = os.path.join(BASE_DIR, "SYNOPSIS_REPORT.html")
PDF_PATH = os.path.join(BASE_DIR, "SYNOPSIS_REPORT.pdf")

with open(MD_PATH, "r", encoding="utf-8") as f:
    raw_md = f.read()

# Separate Title Page from body
parts = raw_md.split("<!-- PAGE BREAK -->", 1)
title_markdown = parts[0].strip()
body_markdown = parts[1].strip() if len(parts) > 1 else ""

# Replace further comment page breaks with HTML page breaks
body_markdown = body_markdown.replace("<!-- PAGE BREAK -->", '<div class="page-break"></div>')

# Use mistune with table plugin
md = mistune.create_markdown(plugins=['table'])
html_title = md(title_markdown)
html_body = md(body_markdown)

html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Synopsis Report - Calories Burnt Prediction</title>
<style>
    @page {{
        size: A4;
        margin: 25mm 20mm 25mm 20mm;
        @bottom-right {{
            content: counter(page);
            font-size: 10pt;
            font-family: 'Times New Roman', Times, serif;
        }}
    }}
    body {{
        font-family: 'Times New Roman', Times, serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #111;
        text-align: justify;
    }}
    .title-page {{
        height: 85vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        page-break-after: always;
        break-after: page;
    }}
    .title-page h1 {{
        font-size: 23pt;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        line-height: 1.45;
        color: #000;
        margin: 0;
    }}
    .page-break {{
        page-break-after: always;
        break-after: page;
    }}
    h1 {{
        font-size: 15pt;
        font-weight: bold;
        color: #111;
        margin-top: 1.6em;
        margin-bottom: 0.6em;
        border-bottom: 1.5px solid #222;
        padding-bottom: 4px;
        page-break-after: avoid;
    }}
    h2 {{
        font-size: 12.5pt;
        font-weight: bold;
        color: #222;
        margin-top: 1.3em;
        margin-bottom: 0.4em;
        page-break-after: avoid;
    }}
    h3 {{
        font-size: 11.5pt;
        font-weight: bold;
        color: #333;
        margin-top: 1.1em;
        margin-bottom: 0.3em;
        page-break-after: avoid;
    }}
    p {{
        margin-top: 0;
        margin-bottom: 0.8em;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 1.2em 0;
        font-size: 9.5pt;
        page-break-inside: avoid;
    }}
    th, td {{
        border: 1px solid #666;
        padding: 6px 9px;
        text-align: left;
    }}
    th {{
        background-color: #f0f0f0;
        font-weight: bold;
        color: #000;
    }}
    tr:nth-child(even) {{
        background-color: #fafafa;
    }}
    blockquote {{
        border-left: 3.5px solid #0056b3;
        background-color: #f7f9fa;
        margin: 1em 0;
        padding: 8px 14px;
        font-style: italic;
        color: #333;
    }}
    code, pre {{
        font-family: 'Courier New', Courier, monospace;
        font-size: 9pt;
        background-color: #f5f5f5;
    }}
    pre {{
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 4px;
        white-space: pre-wrap;
        page-break-inside: avoid;
    }}
    ul, ol {{
        margin-top: 0.3em;
        margin-bottom: 0.8em;
        padding-left: 1.5em;
    }}
    li {{
        margin-bottom: 0.3em;
    }}
    hr {{
        border: 0;
        height: 1px;
        background: #ccc;
        margin: 2em 0;
    }}
</style>
</head>
<body>

<div class="title-page">
    {html_title}
</div>

<div class="report-content">
    {html_body}
</div>

</body>
</html>
"""

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html_document)

# Locate Chrome or Edge
browser_candidates = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
browser_exe = next((p for p in browser_candidates if os.path.exists(p)), None)

if not browser_exe:
    raise RuntimeError("No compatible headless browser (Chrome/Edge) found for PDF conversion.")

print(f"[INFO] Using browser engine: {browser_exe}")
cmd = [
    browser_exe,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={PDF_PATH}",
    HTML_PATH
]

res = subprocess.run(cmd, capture_output=True, text=True)
if os.path.exists(PDF_PATH):
    size_kb = round(os.path.getsize(PDF_PATH) / 1024, 1)
    print(f"[SUCCESS] PDF Generated successfully: {PDF_PATH} ({size_kb} KB)")
else:
    print(f"[ERROR] Failed to generate PDF. Exit code: {res.returncode}, Stderr: {res.stderr}")
