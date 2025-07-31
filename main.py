from agents.parser3 import analyze_repo
from agents.embedder import generate_embeddings, store_embeddings

parsed_file_path = analyze_repo("https://github.com/Mingyue-Cheng/FormerTime")

# embeddings = generate_embeddings(parsed_file_path)
#store_embeddings(embeddings)