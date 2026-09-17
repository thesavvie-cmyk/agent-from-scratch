#!/usr/bin/env python3
"""Extract and print text from a PDF file.

Usage: python read_pdf.py <path_to_pdf> [max_chars]

Requires: pip install pypdf
"""
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: read_pdf.py <file.pdf> [max_chars=5000]")
        sys.exit(1)

    path = sys.argv[1]
    max_chars = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

    try:
        import pypdf
    except ImportError:
        print("pypdf not installed. Run: pip install pypdf")
        sys.exit(1)

    reader = pypdf.PdfReader(path)
    print(f"Pages: {len(reader.pages)}")

    full_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        full_text.append(f"--- Page {i + 1} ---\n{text}")

    combined = "\n".join(full_text)
    print(f"Total chars: {len(combined)}")
    print()
    print(combined[:max_chars])
    if len(combined) > max_chars:
        print(f"\n... (truncated at {max_chars} chars)")


if __name__ == "__main__":
    main()
