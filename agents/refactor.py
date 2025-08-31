from langchain.prompts import ChatPromptTemplate
from llama_cpp import Llama
import json
import os

# === Load the same Llama model you used in Critic ===
llm = Llama.from_pretrained(
    repo_id="TheBloke/CodeLlama-13B-Instruct-GGUF",
    filename="codellama-13b-instruct.Q2_K.gguf",
    n_ctx=8192,
)

# === Load the JSON file with critic responses ===
with open("C:/Users/hp/Desktop/code-refactoring/Multi-Agent-Code-Refactoring-System/oldparsedfiles/Python-Speech-Recognition.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# === Refactor prompt template ===
refactor_prompt = ChatPromptTemplate.from_template("""
[INST] <<SYS>>
You are a senior software engineer specializing in clean code and refactoring.
Take the following function/class along with its critique and produce a **refactored version**.
Make sure the new code is:
- Cleaner and more modular
- Matches Python best practices (PEP8)
- Preserves the original functionality
- Easy to maintain and test
<</SYS>>

--- CONTEXT ---
- **File**: {file}
- **Entity Type**: {type}
- **Entity Name**: {name}
- **Original Critique**: {critique}

--- ORIGINAL CODE ---
{code}

--- TASK ---
Refactor this code. Only output the full improved code (no explanations).
[/INST]
""")

# === Run through each chunk and refactor ===
for chunk in chunks:
    if "llm_response" not in chunk:
        continue  # skip chunks not reviewed by critic

    messages = refactor_prompt.format_messages(
        file=chunk["file"],
        type=chunk["type"],
        name=chunk["name"],
        critique=chunk["llm_response"],
        code=chunk["code"],
    )

    prompt_text = messages[0].content
    print(f"\nRefactoring {chunk['name']}...\n")

    response = llm.create_completion(
        prompt=prompt_text,
        max_tokens=1024,  # more room for full function/class rewrite
        temperature=0.7   # lower temp for more deterministic, clean code
    )

    llm_text = response["choices"][0]["text"].strip()
    print(llm_text)

    # Save refactored code back into JSON
    chunk["refactored_code"] = llm_text

# === Save updated JSON with refactored code ===
with open("parsed files/Python-Speech-Recognition.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)
