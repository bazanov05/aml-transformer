from curl_cffi import requests
import fitz  # PyMuPDF
import os
import re

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
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fsrb-fur/Poland-FUR-MONEYVAL-October-2024.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fsrb-mer/BVI-CFATF-MER-2024.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fur/Follow-Up-Report-United-Kingdom-2022.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fsrb-fur/Moneyval-Follow-Up-Report-Ukraine.pdf.coredownload.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/images/fsrb-mer/MER-MONEYVAL-Ukraine-Dec-2017.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/mer/Mutual-Evaluation-Russian-Federation-2019.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/mer/MER%20Russia%20ful.pdf.coredownload.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fsrb-mer/EAG-Mutual-Evaluation-Report-Belarus-2019.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fsrb-fur/Kyrgyz-Republic%20FUR-2024.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fsrb-mer/Romania-Moneyval-Mutual-Evaluation-2023.pdf.coredownload.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fur/Follow-Up-Report-China-2022.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/mer/Mutual-Evaluation-Report-Germany-2022.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/mer/Mutual-Evaluation-France-2022.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fur/Japan-FUR-2023.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/mer/MER%20Japan%20full.pdf.coredownload.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/fur/Follow-Up-Assessment-Spain-2019.pdf.coredownload.inline.pdf",
    "https://www.fatf-gafi.org/content/dam/fatf-gafi/mer/Mutual-Evaluation-Report-Japan-2021.pdf.coredownload.inline.pdf"
]

seen_paragraphs = set()

with open("aml_corpus.txt", "w", encoding="utf-8") as out_file:
    for url in urls:
        print(f"Downloading: {url}")
        try:
            response = requests.get(url, impersonate="chrome", timeout=30)
            response.raise_for_status()
            
            with open("temp.pdf", "wb") as f:
                f.write(response.content)
            
            doc = fitz.open("temp.pdf")
            
            raw_lines = []
            for page in doc:
                raw_lines.extend(page.get_text().split('\n'))
            doc.close()
            
            valid_lines = []
            for line in raw_lines:
                line = line.strip()
                if not line:
                    valid_lines.append("")
                    continue
                if line.isdigit():
                    continue
                if len(line) < 5:
                    continue
                if re.search(r'\.{4,}', line):
                    continue
                valid_lines.append(line)
            
            paragraphs = []
            current_para = ""
            for line in valid_lines:
                if line == "":
                    if current_para:
                        paragraphs.append(re.sub(r'\s+', ' ', current_para).strip())
                        current_para = ""
                else:
                    if current_para:
                        if current_para.endswith('-'):
                            current_para = current_para[:-1] + line
                        else:
                            current_para += " " + line
                    else:
                        current_para = line
                        
            if current_para:
                paragraphs.append(re.sub(r'\s+', ' ', current_para).strip())
                
            out_file.write("<|document|>\n\n")
            for p in paragraphs:
                if p not in seen_paragraphs:
                    seen_paragraphs.add(p)
                    out_file.write(p + "\n\n")
                    
        except Exception as e:
            print(f"Failed to process {url}: {e}")

if os.path.exists("temp.pdf"):
    os.remove("temp.pdf")

print(f"Done. File size: {os.path.getsize('aml_corpus.txt') / (1024 * 1024):.2f} MB")