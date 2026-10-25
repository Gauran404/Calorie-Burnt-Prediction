import os
import subprocess
import pypdf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE_DIR, "RBU_PROJECT_SYNOPSIS.html")
PDF_PATH = os.path.join(BASE_DIR, "RBU_PROJECT_SYNOPSIS.pdf")

browser_candidates = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
browser_exe = next((p for p in browser_candidates if os.path.exists(p)), None)

if not browser_exe:
    raise RuntimeError("No compatible browser found.")

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
    reader = pypdf.PdfReader(PDF_PATH)
    page_count = len(reader.pages)
    print(f"[SUCCESS] PDF Generated: {PDF_PATH} ({size_kb} KB)")
    print(f"[PAGE COUNT] Generated PDF has {page_count} pages.")
    for i, page in enumerate(reader.pages):
        lines = [l.strip() for l in page.extract_text().splitlines() if l.strip()]
        first = lines[0] if lines else "EMPTY"
        last = lines[-1] if lines else "EMPTY"
        print(f"  Page {i+1}: length={len(page.extract_text()):4d} chars | Starts with: {first[:40]} | Ends with: {last[:40]}")
else:
    print(f"[ERROR] PDF generation failed: {res.stderr}")
