from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from llama_cpp import Llama
import os
import json

llm = Llama.from_pretrained(
	repo_id="TheBloke/CodeLlama-13B-Instruct-GGUF",
	filename="codellama-13b-instruct.Q2_K.gguf",
    n_ctx=8192,
)
with open("parsed files/Python-Speech-Recognition.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

def heuristics_filter(metrics):
    return (
        metrics["cyclomatic_complexity"] > 10 or
        metrics["nesting_depth"] > 3 or
        metrics["line_count"] > 50 or
        not metrics.get("has_docstring", False) or 
        len(metrics["magic_numbers"]) >3
    )

prompt_template = ChatPromptTemplate.from_template("""
[INST] <<SYS>>
You are a seasoned Software Refactoring Critic. Your job is to identify all code smells, anti-patterns, and design inefficiencies, and provide actionable, high-quality suggestions that align with modern software engineering best practices.
<</SYS>>

--- CONTEXT ---
- **File**: {file}
- **Entity Type**: {type}
- **Entity Name**: {name}
- **Line Count**: {line_count}
- **Cyclomatic Complexity**: {cyclomatic_complexity}
- **Nesting Depth**: {nesting_depth}
- **Has Docstring**: {has_docstring}
- **Magic Numbers Count**: {magic_numbers}
- **Comments**: {comments}
- **Dependencies**: {dependencies}
- **Imports**: {imports}

--- SOURCE CODE ---
{code}

--- TASK INSTRUCTIONS ---
Carefully review the code and answer the following:
1. **Code Smells**: Identify and explain all present code smells (e.g., long method, large class, magic numbers, deep nesting, data clumps, low cohesion, high coupling).
2. **Refactoring Opportunities**: Point out design flaws and suggest clear, specific refactorings (e.g., extract method/class, reduce nesting, improve naming, decouple responsibilities).
3. **Documentation & Style**: Comment on the presence and quality of docstrings, comments, naming conventions, and adherence to style guides.
4. **Maintainability**: Evaluate how maintainable, readable, and testable the code is. Propose ways to improve it.
5. **Summary Recommendation**: Conclude with an overall refactoring priority (low/medium/high) and a rationale.

Respond in a structured and concise format with bullet points where helpful.
[/INST]
""")

for chunk in chunks:
    metrics = chunk["code_metrics"]
    if heuristics_filter(metrics):
        messages = prompt_template.format_messages(
            file = chunk["file"],
            type = chunk["type"],
            name = chunk["name"],
            line_count=metrics["line_count"],
            cyclomatic_complexity=metrics["cyclomatic_complexity"],
            nesting_depth=metrics["nesting_depth"],
            has_docstring=metrics["has_docstring"],
            magic_numbers=metrics["magic_numbers"],
            code=chunk["code"],
            comments = chunk["comments"],
            dependencies = chunk["dependencies"],
            imports = chunk["imports"]
        )
        prompt_text = messages[0].content
        print(f"\nCritiquing {chunk['name']}...\n")
        print(prompt_text)
        response = llm.create_completion(
        prompt=prompt_text,
        max_tokens=512,   # enough room for a full answer
        temperature=0.7
        )
        print(response)
        llm_text = response["choices"][0]["text"]
        print(llm_text)
        chunk["llm_response"] = llm_text.strip()
with open("parsed files/Python-Speech-Recognition.json", "w") as f:
    json.dump(chunks, f, indent=2)



    




