import ollama
from ollama import Client
from src.rag import Rag
from typing import Optional
from src.config import SYSTEM_PROMPT, MODEL_LLM, HOST_LLM


class RAGChatbot:
    def __init__(self):
        self.client = Client(host=HOST_LLM) if HOST_LLM else ollama.Client()
        self.rag = Rag()
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.max_history_length = 10
    
    def add_to_history(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})
        
        # Limit history to prevent token bloat
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def chat(self, user_input: str) -> Optional[str]:
        """Process user input and return assistant response"""
        self.add_to_history("user", user_input)
        
        data = self.rag.retrieve_data(user_input)
        prompt_text = \
            (
                "Nutze die folgenden Informationen zur Beantwortung der Frage. "
                f"Kontext:\n{data}\n\n"
                f"Frage:\n{user_input}"
            )
        
        # Build messages with conversation history and current prompt
        messages = self.conversation_history + [{"role": "user", "content": f"{prompt_text}"}]
        
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
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        print("Conversation history cleared.")
