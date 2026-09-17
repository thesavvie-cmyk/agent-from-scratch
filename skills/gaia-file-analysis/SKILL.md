---
name: gaia-file-analysis
description: Analyze GAIA task attachments — Excel spreadsheets, CSV files, PDFs, and images — by uploading them to the sandbox and processing with execute_python
version: "1.0"
scripts:
  - scripts/read_excel.py
  - scripts/read_pdf.py
---

# GAIA File Analysis

GAIA tasks sometimes include file attachments (Excel, CSV, PDF, image).
The attachment path is available in the task metadata as `file_path`.
The correct workflow is: **upload → execute_python → read result**.

---

## Upload workflow

```python
# 1. Read the file locally
with open(task["file_path"], "rb") as f:
    raw = f.read()

# 2. Upload to sandbox
await sandbox.files.write("/home/user/attachment.xlsx", raw)

# 3. Process in execute_python
```

For text files (CSV, plain text) you can also use `write_sandbox_file` with
the decoded content. For binary files (Excel, PDF, images) upload the raw bytes.

---

## Excel / CSV

**openpyxl** (xlsx) is pre-installed. Use it directly:

```python
import openpyxl
wb = openpyxl.load_workbook('/home/user/attachment.xlsx')
ws = wb.active

# Print all rows
for row in ws.iter_rows(values_only=True):
    print(row)

# Find a value
headers = [cell.value for cell in ws[1]]
print("Columns:", headers)
```

For CSV:
```python
import csv
with open('/home/user/attachment.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print(f"{len(rows)} rows, columns: {list(rows[0].keys())}")
```

**Never count rows by reading text** — always use `ws.max_row` or `len(rows)`.

---

## PDF

PDFs in GAIA are usually text-based (not scanned). Install pypdf first:

```python
# Step 1 (run_command):
# pip install pypdf --quiet

# Step 2 (execute_python):
import pypdf
reader = pypdf.PdfReader('/home/user/attachment.pdf')
text = '\n'.join(page.extract_text() or '' for page in reader.pages)
print(f"Pages: {len(reader.pages)}, chars: {len(text)}")
print(text[:3000])
```

If `extract_text()` returns empty strings, the PDF is scanned — try
`pdfminer.six` or note that OCR would be needed.

---

## Images

Images may already be included in the task prompt (vision model input).
If you need to inspect metadata or dimensions:

```python
# After: pip install pillow (run_command)
from PIL import Image
img = Image.open('/home/user/attachment.png')
print(f"Size: {img.size}, Mode: {img.mode}")
```

For counting objects or reading text in an image, pass the image path to the
model's vision capability (include the image in the request, not in code).

---

## Key rules

- **Count programmatically** — never try to count rows/columns from a text description
- **Filter in code** — for "how many rows where column X = Y", use a list comprehension
- **Print partial output** — always `print(text[:3000])` first to confirm parsing worked
- **Check for empty results** — a zero-row read usually means a wrong sheet or encoding

---

## Scripts in this skill

| Script | Purpose |
|--------|---------|
| `scripts/read_excel.py` | Print all sheets, headers, and first 5 rows of an xlsx |
| `scripts/read_pdf.py` | Extract and print text from a PDF file |

Run them in the sandbox: `run_command("python /home/user/skills/gaia-file-analysis/scripts/read_excel.py /home/user/attachment.xlsx")`
