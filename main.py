import ollama

response = ollama.chat(
    model="qwen3:0.6b",
    messages=[
        {"role": "user", "content": "Erkläre kurz, was ein LLM ist."}
    ]
)

print(response["message"]["content"])
