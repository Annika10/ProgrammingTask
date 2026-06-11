import os
from ollama import Client
from src.rag import Rag
from typing import Optional
from src.config import SYSTEM_PROMPT, MODEL_LLM, LOCAL_MODEL_LLM
from dotenv import load_dotenv


class LLMChatbot:
    def __init__(self, local: bool = False) -> None:
        """
        initialize LLM Chatbot
        Args:
            local (bool): determines if a local LLM or cloud LLM is used
        """
        load_dotenv()
        self.api_key = os.getenv("OLLAMA_API_KEY")
        if local:
            self.client = Client()
            self.model = LOCAL_MODEL_LLM
        else:
            self.client = Client(host="https://ollama.com", headers={
                'Authorization': 'Bearer ' + self.api_key}) if self.api_key else Client()
            self.model = MODEL_LLM
        self.rag = Rag()
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.max_history_length = 10
    
    def add_to_history(self, role: str, content: str) -> None:
        """
        add current conversation input in history and maintain history lengths
        Args:
            role (str): role of conversation input (either system, assistant, or user)
            content (str): content of conversation part
        Returns: None
        """
        self.conversation_history.append({"role": role, "content": content})
        
        # Limit history to prevent token bloat
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def chat(self, user_input: str) -> Optional[str]:
        """
        Process user input and return assistant response
        Args:
            user_input (str): current user input

        Returns:
            Optional[str]: assistant response or None if error occurs
        """
        self.add_to_history("user", user_input)
        
        data = self.rag.retrieve_data(user_input)
        print("data")
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
                model=self.model,
                messages=messages,
                stream=True,
                think=False
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
