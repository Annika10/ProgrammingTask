from rag_chatbot import RAGChatbot


def main():
    chatbot = RAGChatbot()
    
    print("\n" + "=" * 60)
    print("🤖 RAG Chatbot")
    print("=" * 60)
    print("Stelle Fragen zu Beckhoff.")
    print("Commands: 'history' - Gesprächsverlauf zeigen | 'clear' - History löschen | 'quit' - Beenden")
    print("=" * 60 + "\n")

    while True:
        user_input = input("📝 Du: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit']:
            print("Auf Wiedersehen!")
            break
        
        if user_input.lower() == 'clear':
            chatbot.clear_history()
            continue
        
        if user_input.lower() == 'history':
            if chatbot.conversation_history:
                print("\n📋 Conversation history:\n:")
                for i, msg in enumerate(chatbot.conversation_history, 1):
                    role_display = "Du" if msg["role"] == "user" else "🤖 Bot"
                    print(f"{i}. {role_display}: {msg['content'][:100]}...")
                print()
            else:
                print("No history by now.\n")
            continue
        
        response = chatbot.chat(user_input)
        if response:
            print(f"\n🤖 Bot: {response}\n")


if __name__ == "__main__":
    main()
