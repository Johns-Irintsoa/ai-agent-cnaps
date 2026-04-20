from langchain_openai import ChatOpenAI


def main ():
    llm = ChatOpenAI(
        model="huggingface.co/bartowski/mistral-7b-instruct-v0.3-gguf:q4_k_m",             # Use the exact local model ID
        base_url="http://localhost:12434/engines/llama.cpp/v1/chat/completions", # Default DMR TCP endpoint
        api_key="not-needed"                       # DMR ignores this but LangChain requires it
    )

    response = llm.invoke("What is Docker Model Runner?")
    print(response.content)

if __name__ == "__main__":
    main()