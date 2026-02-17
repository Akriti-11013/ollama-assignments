from langchain_ollama import OllamaLLM

# Load smallest & fastest model
llm = OllamaLLM(model="phi3")

print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = llm.invoke(user_input)
    print("LLM:", response)
