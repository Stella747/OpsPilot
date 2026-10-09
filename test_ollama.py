
import ollama


response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "user",
            "content": (
                "Explain database connection pool exhaustion "
                "in simple terms in 3 sentences."
            )
        }
    ]
)


print("Qwen's response:")
print()
print(response["message"]["content"])
