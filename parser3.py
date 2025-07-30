import os
import shutil
import ast
import json
import re
from pathlib import Path
from git import Repo
from collections import defaultdict

# --- Utility Functions ---

def annotate_ast_with_parents(tree):
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent

def get_ast_path(node):
    path = []
    while node:
        if isinstance(node, ast.Module):
            path.insert(0, "Module")
            break
        elif isinstance(node, ast.ClassDef):
            path.insert(0, f"ClassDef:{node.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            path.insert(0, f"{type(node).__name__}:{node.name}")
        node = getattr(node, 'parent', None)
    return path or ["Module"]

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
    return sorted(list(calls))

def extract_imports(tree):
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module if node.module else ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    return sorted(set(imports))

def get_cyclomatic_complexity(node):
    count = 1  # start with 1 for the function itself
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp)):  # branching structures
            count += 1
    return count

def get_nesting_depth(node):
    max_depth = [0]
    def helper(n, depth):
        if isinstance(n, (ast.If, ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef)):
            depth += 1
        max_depth[0] = max(max_depth[0], depth)
        for child in ast.iter_child_nodes(n):
            helper(child, depth)
    helper(node, 0)
    return max_depth[0]

def extract_magic_numbers(node):
    magic_numbers = set()
    class MagicNumberVisitor(ast.NodeVisitor):
        def visit_Constant(self, n):
            if isinstance(n.value, (int, float)) and n.value not in (0, 1):
                magic_numbers.add(n.value)
    MagicNumberVisitor().visit(node)
    return sorted(list(magic_numbers))

def parse_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    source_lines = source.splitlines(keepends=True)

    try:
        tree = ast.parse(source)
        annotate_ast_with_parents(tree)
    except SyntaxError:
        return []

    imports = extract_imports(tree)
    chunks = []
    module_path = filepath.replace("\\", "/")
    class_name = None
    found_code_chunk = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            try:
                is_method = any(isinstance(p, ast.ClassDef) for p in ast.iter_parent(node)) if hasattr(ast, 'iter_parent') else bool(class_name)
                decorators = [ast.unparse(d) for d in node.decorator_list] if hasattr(node, 'decorator_list') else []
                args_count = len(node.args.args) if hasattr(node, 'args') else 0
                return_type = ast.unparse(node.returns) if hasattr(node, 'returns') and node.returns else None
                docstring = ast.get_docstring(node)
                code = extract_code(source_lines, node)
                deps = extract_dependencies(node)

                chunk = {
                    "id": f"{module_path}::{node.name}",
                    "file": module_path,
                    "name": node.name,
                    "type": type(node).__name__,
                    "language": "python",
                    "ast_path": get_ast_path(node),
                    "is_script_entry": False,
                    "context": {
                        "class": class_name if class_name else None,
                        "is_method": is_method,
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "decorators": decorators,
                        "args_count": args_count,
                        "returns": return_type
                    },
                    "code_metrics": {
                        "line_count": len(code.splitlines()),
                        "nesting_depth": get_nesting_depth(node),
                        "cyclomatic_complexity": get_cyclomatic_complexity(node),
                        "magic_numbers": extract_magic_numbers(node),
                        "has_docstring": bool(docstring)
                    },
                    "docstring": docstring if docstring else "",
                    "code": code,
                    "dependencies": deps,
                    "imports": imports
                }
                chunks.append(chunk)
                found_code_chunk=True
            except Exception as e:
                print(f"⚠️ Failed on node in {filepath}: {e}")
    # Add top-level script block if no functions/classes found
    if not found_code_chunk and source.strip():
        chunk = {
            "id": f"{module_path}::__top_level__",
            "file": module_path,
            "name": "__top_level__",
            "type": "TopLevel",
            "language": "python",
            "ast_path": ["Module"],
            "is_script_entry": True,
            "context": {
                "class": None,
                "is_method": False,
                "is_async": False,
                "decorators": [],
                "args_count": 0,
                "returns": None
            },
            "code_metrics": {
                "line_count": len(source_lines),
                "nesting_depth": 0,
                "cyclomatic_complexity": 0,
                "magic_numbers": extract_magic_numbers(tree),
                "has_docstring": False
            },
            "docstring": "",
            "code": source,
            "dependencies": extract_dependencies(tree),
            "imports": imports
        }
        chunks.append(chunk)

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

def analyze_repo(repo_url, output_file="parsed_repo_3.json"):
    clone_path = "repo-clone"
    if os.path.exists(clone_path):
        shutil.rmtree(clone_path)
    Repo.clone_from(repo_url, clone_path)
    print(f"Cloned repo: {repo_url}")

    chunks = walk_repo(clone_path)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Extracted and saved {len(chunks)} code chunks to {output_file}")

# Example usage
analyze_repo("https://github.com/Kalebu/Python-Speech-Recognition-.git")
