from curl_cffi import requests
import fitz  # PyMuPDF
import os

urls = [
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/mer/MER%20US%20full.pdf.coredownload.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fur/USA-FUR-2024.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fsrb-fur/bulgaria-fur-2026.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fur/indonesia-fur-2026.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/mer/MER-Italy-2026.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/mer/MER-Austria-2026.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fsrb-mer/Maldives-MER-2025.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fsrb-fur/Estonia-FUR-2025.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fsrb-mer/Serbia-MER-2025.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fsrb-fur/Montenegro-FUR-2025.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fur/Netherlands-Follow-Up-Report-2025.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fsrb-fur/Poland-FUR-MONEYVAL-October-2024.pdf.coredownload.inline.pdf"
]

def clean_text(text):
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.isdigit():  # Strip isolated page numbers
            continue
        if len(line) < 5:   # Strip very short headers/footers
            continue
        cleaned_lines.append(line)
    
    return " ".join(cleaned_lines)


with open("aml_corpus.txt", "w", encoding="utf-8") as out_file:
    for url in urls:
        print(f"Downloading: {url}")
        try:
            response = requests.get(url, impersonate="chrome", timeout=30)
            response.raise_for_status()
            
            with open("temp.pdf", "wb") as f:
                f.write(response.content)
            
            doc = fitz.open("temp.pdf")
            for page in doc:
                raw_text = page.get_text()
                cleaned_page = clean_text(raw_text)
                if cleaned_page:
                    out_file.write(cleaned_page + "\n")
            doc.close()
            
        except Exception as e:
            print(f"Failed to process {url}: {e}")


if os.path.exists("temp.pdf"):
    os.remove("temp.pdf")
    

print(f"Done. File size: {os.path.getsize('aml_corpus.txt') / (1024 * 1024):.2f} MB")