from agents.parser3 import analyze_repo
from agents.embedder import generate_embeddings, store_embeddings

analyze_repo("https://github.com/Kalebu/Python-Speech-Recognition-.git")

# embeddings = generate_embeddings(parsed_file_path)
#store_embeddings(embeddings)