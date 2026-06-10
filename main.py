import argparse
from src.rag_chatbot import RAGChatbot


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Chatbot")
    parser.add_argument("--local", action="store_true",
                        help="Use local LLM instead of cloud")
    args = parser.parse_args()
    
    chatbot = RAGChatbot(local=args.local)
    
    print("\n" + "=" * 60)
    print("🤖 RAG Chatbot")
    print("=" * 60)
    print("Stelle Fragen zu Beckhoff.")
    print("Commands: 'quit' - Beenden")
    print("=" * 60 + "\n")
    
    while True:
        user_input = input("📝 Du: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit']:
            print("Auf Wiedersehen!")
            break
        
        print(f"🤖 Bot: ", end="")
        response = chatbot.chat(user_input)


if __name__ == "__main__":
    main()
