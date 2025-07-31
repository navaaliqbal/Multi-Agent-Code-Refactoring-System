import json
from tqdm import tqdm

def generate_embeddings(parsed_file = "parsed_repo.json", index_file="code_index.faiss"):
    with open(parsed_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    for chunk in tqdm(chunks):
        text = chunk["code"]
        print(text)


generate_embeddings()