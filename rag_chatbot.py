import ollama
from ollama import Client
from rag import Rag
from typing import Optional
from config import SYSTEM_PROMPT, MODEL_LLM, HOST_LLM

class RAGChatbot:
    def __init__(self):
        self.client = Client(host=HOST_LLM) if HOST_LLM else ollama.Client()
        self.rag = Rag()
        self.conversation_history = []
        self.max_history_length = 10
    
    def add_to_history(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})
        
        # Limit history to prevent token bloat
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def get_message_context(self, user_input: str) -> str:
        data = self.rag.retrieve_data(user_input)
        return (
            "Nutze die folgenden Informationen zur Beantwortung der Frage:\n\n"
            f"Kontext:\n{data}"
        )
    
    def chat(self, user_input: str) -> Optional[str]:
        """Process user input and return assistant response"""
        self.add_to_history("user", user_input)
        
        context = self.get_message_context(user_input)
        
        # Build messages with system prompt and conversation history
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nFrage:\n{user_input}"}
        ]
        
        # Add recent history for context (excluding the current message)
        if len(self.conversation_history) > 1:
            for msg in self.conversation_history[:-1]:
                if msg["role"] != "user":  # Don't duplicate current user input
                    messages.append(msg)
        
        try:
            full_response = ""
            response = self.client.chat(
                model=MODEL_LLM,
                messages=messages,
                stream=True
            )
            
            for chunk in response:
                content = chunk["message"]["content"]
                print(content, end="", flush=True)
                full_response += content
            
            print()  # new line after answer
            self.add_to_history("assistant", full_response)
            return full_response
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared.")