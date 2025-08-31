import json
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import ChatPromptTemplate

# === Load LLM ===
print("Loading CodeLlama model...")
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-13b-Instruct-hf")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-13b-Instruct-hf")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)
llm = HuggingFacePipeline(pipeline=pipe)

# === Prompt Template ===
prompt_template = ChatPromptTemplate.from_template("""
You are a seasoned Software Refactoring Critic tasked with identifying all code smells, anti-patterns, and design inefficiencies in the following code. Your goal is to provide actionable, high-quality suggestions that align with modern software engineering best practices.

--- CONTEXT ---
- **File**: {file}
- **Entity Type**: {type}
- **Entity Names**: {name}
- **Combined Line Count**: {line_count}
- **Max Cyclomatic Complexity**: {cyclomatic_complexity}
- **Max Nesting Depth**: {nesting_depth}
- **All Have Docstrings**: {has_docstring}
- **Magic Numbers Count**: {magic_numbers}
- **Comments**: {comments}
- **Dependencies**: {dependencies}
- **Imports**: {imports}

--- SOURCE CODE ---
{code}

--- TASK INSTRUCTIONS ---
Carefully review the code and answer the following:
1. **Code Smells**: Identify and explain all present code smells.
2. **Refactoring Opportunities**: Suggest clear, specific refactorings.
3. **Documentation & Style**: Evaluate comments, naming, and style.
4. **Maintainability**: Assess readability, testability, modularity.
5. **Summary Recommendation**: Give refactoring priority (low/medium/high) with rationale.

Respond in a structured and concise format with bullet points where helpful.
""")

# === Heuristic Rules ===
def run_heuristics(metrics):
    issues = []
    if metrics["line_count"] > 200:
        issues.append("🔸 Long file: over 200 lines.")
    if metrics["cyclomatic_complexity"] > 10:
        issues.append("🔸 High cyclomatic complexity.")
    if metrics["nesting_depth"] > 4:
        issues.append("🔸 Deep nesting levels.")
    if not metrics["has_docstring"]:
        issues.append("🔸 Missing docstrings.")
    if len(metrics["magic_numbers"]) > 3:
        issues.append("🔸 Excessive use of magic numbers.")
    return issues

# === Merge Metrics Across Chunks ===
def merge_metrics(chunks):
    return {
        "line_count": sum(c["code_metrics"].get("line_count", 0) for c in chunks),
        "cyclomatic_complexity": max((c["code_metrics"].get("cyclomatic_complexity", 0) for c in chunks), default=0),
        "nesting_depth": max((c["code_metrics"].get("nesting_depth", 0) for c in chunks), default=0),
        "has_docstring": all(c["code_metrics"].get("has_docstring", False) for c in chunks),
        "magic_numbers": sum((c["code_metrics"].get("magic_numbers", []) for c in chunks), []),
        "comments": sum((c.get("comments", []) for c in chunks), []),
        "dependencies": sum((c.get("dependencies", []) for c in chunks), []),
        "imports": sum((c.get("imports", []) for c in chunks), []),
    }

# === Main Function ===
def critic_main_grouped(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    grouped_by_file = defaultdict(list)
    for chunk in data:
        grouped_by_file[chunk["file"]].append(chunk)

    results = []

    for file, chunks in grouped_by_file.items():
        print(f"\n📂 Critiquing file: {file} ({len(chunks)} code entities)")
        total_code = "\n\n".join(c.get("code", "") for c in chunks)
        metric_summary = merge_metrics(chunks)
        heuristics = run_heuristics(metric_summary)

        if not heuristics:
            print("⚠️  Skipping file (no heuristics triggered)")
            continue  # Skip if heuristics don't indicate issues

        try:
            messages = prompt_template.format_messages(
                file=file,
                type="grouped_entities",
                name=", ".join(c.get("name", "unknown") for c in chunks),
                line_count=metric_summary["line_count"],
                cyclomatic_complexity=metric_summary["cyclomatic_complexity"],
                nesting_depth=metric_summary["nesting_depth"],
                has_docstring=metric_summary["has_docstring"],
                magic_numbers=metric_summary["magic_numbers"],
                comments=metric_summary["comments"],
                dependencies=metric_summary["dependencies"],
                imports=metric_summary["imports"],
                code=total_code
            )
            prompt_text = messages[0].content
            response = llm(prompt_text)

            results.append({
                "file": file,
                "chunks": [c.get("name") for c in chunks],
                "heuristics": heuristics,
                "llm_response": response.strip()
            })

        except Exception as e:
            results.append({
                "file": file,
                "chunks": [c.get("name") for c in chunks],
                "heuristics": heuristics,
                "llm_response": f"❌ LLM Error: {str(e)}"
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Grouped critique complete. Results saved to: {output_path}")

# === CLI Entry Point ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSON file from parser agent")
    parser.add_argument("--output", type=str, default="critic_output_grouped.json", help="Output JSON file")
    args = parser.parse_args()
    critic_main_grouped(args.input, args.output)
