from huggingface_hub import hf_hub_download
import zipfile
import os

print("Downloading dataset zip...")
file_path = hf_hub_download(repo_id="takala/financial_phrasebank", filename="data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
print(f"Downloaded to {file_path}")

with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall(".")
    
with open("FinancialPhraseBank-v1.0/Sentences_50Agree.txt", "r", encoding="iso-8859-1") as f:
    lines = f.readlines()
    print(lines[:5])
