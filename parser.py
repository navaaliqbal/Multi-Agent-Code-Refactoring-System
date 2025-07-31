import os
import shutil
import ast
import json
from git import Repo
from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")


def extract_code(source_lines, node):
    return "".join(source_lines[node.lineno - 1: node.end_lineno])

def extract_dependencies(node):
    calls = set()
    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, call_node):
            try:
                if isinstance(call_node.func, ast.Attribute):
                    calls.add(ast.unparse(call_node.func))
                elif isinstance(call_node.func, ast.Name):
                    calls.add(call_node.func.id)
            except:
                pass
            self.generic_visit(call_node)

    CallVisitor().visit(node)
    return list(calls)

def generate_embeddings(parsed_file = "parsed_repo.json", index_file="code_index.faiss"):
    with open(parsed_file, "r", encoding="utf-8"):
        chunks = json.load(parsed_file)
    print()


def parse_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    source_lines = source.splitlines(keepends=True)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            try:
                code = extract_code(source_lines, node)
                docstring = ast.get_docstring(node)
                deps = extract_dependencies(node)

                chunks.append({
                    "file": filepath,
                    "name": node.name,
                    "type": type(node).__name__,
                    "docstring": docstring,
                    "code": code,
                    "dependencies": deps,
                })
            except Exception as e:
                print(f"Failed on node in {filepath}: {e}")
    return chunks

def walk_repo(repo_path):
    results = []
    for dirpath, _, filenames in os.walk(repo_path):
        for fname in filenames:
            if fname.endswith(".py"):
                fpath = os.path.join(dirpath, fname)
                chunks = parse_file(fpath)
                results.extend(chunks)
    return results

def analyze_repo(repo_url, output_file="parsed_repo.json"):
    clone_path = "repo-clone"
    if os.path.exists(clone_path):
        shutil.rmtree(clone_path)
    Repo.clone_from(repo_url, clone_path)
    print(f"Cloned repo: {repo_url}")

    chunks = walk_repo(clone_path)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Extracted and saved {len(chunks)} code chunks to {output_file}")


analyze_repo("https://github.com/Mingyue-Cheng/FormerTime")
