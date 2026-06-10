import pytest
from unittest.mock import Mock, patch
from src.llm_chatbot import LLMChatbot
from src.config import SYSTEM_PROMPT


class TestLLMChatbot:
    """Tests for LLMChatbot class"""
    
    @pytest.fixture
    def mock_rag(self):
        """Mock the Rag class to avoid loading actual data"""
        with patch('src.llm_chatbot.Rag') as mock:
            yield mock
    
    @pytest.fixture
    def mock_ollama_client(self):
        """Mock the Ollama client"""
        with patch('src.llm_chatbot.Client') as mock:
            yield mock
    
    @pytest.fixture
    def chatbot(self, mock_rag, mock_ollama_client):
        """Create a LLMChatbot instance with mocks"""
        mock_rag_instance = Mock()
        mock_rag_instance.retrieve_data.return_value = "Test context data"
        mock_rag.return_value = mock_rag_instance
        
        mock_client_instance = Mock()
        mock_ollama_client.return_value = mock_client_instance
        
        return LLMChatbot()
    
    def test_chatbot_initialization(self, chatbot):
        """Test that chatbot initializes with correct attributes"""
        assert chatbot.conversation_history == [{"role": "system", "content": SYSTEM_PROMPT}]
        assert chatbot.max_history_length == 10
        assert chatbot.rag is not None
        assert chatbot.client is not None
    
    def test_add_to_history_user_message(self, chatbot):
        """Test adding user message to history"""
        chatbot.add_to_history("user", "Hello")
        
        assert len(chatbot.conversation_history) == 2
        assert chatbot.conversation_history[1]["role"] == "user"
        assert chatbot.conversation_history[1]["content"] == "Hello"
    
    def test_add_to_history_assistant_message(self, chatbot):
        """Test adding assistant message to history"""
        chatbot.add_to_history("assistant", "Hi there!")
        
        assert len(chatbot.conversation_history) == 2
        assert chatbot.conversation_history[1]["role"] == "assistant"
        assert chatbot.conversation_history[1]["content"] == "Hi there!"
    
    def test_history_limit_enforcement(self, chatbot):
        """Test that conversation history respects max_history_length"""
        for i in range(15):
            chatbot.add_to_history("user", f"Message {i}")
        
        assert len(chatbot.conversation_history) == 10
        assert chatbot.conversation_history[0]["content"] == "Message 5"
        assert chatbot.conversation_history[-1]["content"] == "Message 14"
    
    def test_chat_stream_response(self, chatbot):
        """Test chat method with streaming response"""
        mock_response = [
            {"message": {"content": "Hello"}},
            {"message": {"content": " there"}},
            {"message": {"content": "!"}}
        ]
        chatbot.client.chat.return_value = mock_response
        
        with patch('builtins.print'):  # Suppress print output
            response = chatbot.chat("Test question")
        
        assert response == "Hello there!"
        assert len(chatbot.conversation_history) == 3  # system + user + assistant
    
    def test_chat_adds_messages_to_history(self, chatbot):
        """Test that chat adds both user and assistant messages to history"""
        chatbot.client.chat.return_value = [
            {"message": {"content": "Response"}}
        ]
        
        with patch('builtins.print'):
            chatbot.chat("Question")
        
        assert len(chatbot.conversation_history) == 3
        assert chatbot.conversation_history[1]["role"] == "user"
        assert chatbot.conversation_history[2]["role"] == "assistant"
    
    def test_chat_error_handling(self, chatbot):
        """Test that chat handles exceptions gracefully"""
        chatbot.client.chat.side_effect = Exception("Connection error")
        
        with patch('builtins.print'):
            response = chatbot.chat("Question")
        
        assert response is None
    
    def test_clear_history(self, chatbot):
        """Test clearing conversation history"""
        chatbot.add_to_history("user", "Message 1")
        chatbot.add_to_history("assistant", "Response 1")
        
        assert len(chatbot.conversation_history) == 3
        
        with patch('builtins.print'):
            chatbot.clear_history()
        
        assert len(chatbot.conversation_history) == 1
    
    def test_chat_system_prompt_included(self, chatbot):
        """Test that system prompt is included in messages"""
        chatbot.client.chat.return_value = [
            {"message": {"content": "Response"}}
        ]
        
        with patch('builtins.print'):
            chatbot.chat("Test question")
        
        call_args = chatbot.client.chat.call_args
        messages = call_args[1]['messages']
        
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT
    
    def test_chat_correct_model_used(self, chatbot):
        """Test that chat uses correct model"""
        chatbot.client.chat.return_value = [
            {"message": {"content": "Response"}}
        ]
        
        with patch('builtins.print'):
            chatbot.chat("Test")
        
        call_args = chatbot.client.chat.call_args
        assert call_args[1]['model'] == 'gemma4:31b-cloud'


class TestLLMChatbotIntegration:
    """Integration tests for LLMChatbot"""
    
    @pytest.fixture
    def mock_rag(self):
        with patch('src.llm_chatbot.Rag') as mock:
            mock_instance = Mock()
            mock_instance.retrieve_data.return_value = "Beckhoff is a technology company"
            mock.return_value = mock_instance
            yield mock
    
    @pytest.fixture
    def mock_client(self):
        with patch('src.llm_chatbot.Client') as mock:
            yield mock
    
    @pytest.fixture
    def chatbot(self, mock_rag, mock_client):
        return LLMChatbot()
    
    def test_multiple_messages_conversation(self, chatbot):
        """Test a multi-turn conversation"""
        responses = [
            [{"message": {"content": "Response 1"}}],
            [{"message": {"content": "Response 2"}}],
            [{"message": {"content": "Response 3"}}]
        ]
        chatbot.client.chat.side_effect = responses
        
        with patch('builtins.print'):
            chatbot.chat("Question 1")
            chatbot.chat("Question 2")
            chatbot.chat("Question 3")
        
        assert len(chatbot.conversation_history) == 7  # system + 3 user + 3 assistant
    
    def test_history_includes_previous_context(self, chatbot):
        """Test that previous messages are included in context"""
        chatbot.client.chat.return_value = [
            {"message": {"content": "Response"}}
        ]
        
        with patch('builtins.print'):
            chatbot.chat("First question")
            chatbot.chat("Second question")
        
        assert len(chatbot.conversation_history) == 5


class TestOllamaConfiguration:
    """Tests for Ollama client configuration and connectivity"""
    
    def test_ollama_client_with_cloud_mode(self):
        """Test that cloud mode uses ollama.Client when no API key"""
        with patch('src.llm_chatbot.Rag'):
            with patch('src.llm_chatbot.Client') as mock_default_client:
                with patch('src.llm_chatbot.os.getenv', return_value=None):
                    LLMChatbot(local=False)
                    
                    mock_default_client.assert_called_once_with()


class TestOllamaIntegration:
    """Integration tests checking Ollama configuration usage"""
    
    @pytest.fixture
    def chatbot_with_mocks(self):
        """Create a chatbot with properly mocked Ollama client"""
        with patch('src.llm_chatbot.Rag') as mock_rag_class:
            mock_rag = Mock()
            mock_rag.retrieve_data.return_value = "Test context"
            mock_rag_class.return_value = mock_rag
            
            with patch('src.llm_chatbot.Client') as mock_client_class:
                mock_client = Mock()
                mock_client.chat.return_value = [{"message": {"content": "Response"}}]
                mock_client_class.return_value = mock_client
                
                chatbot = LLMChatbot()
                chatbot.client = mock_client
                yield chatbot, mock_client
    
    def test_ollama_receives_model_parameter(self, chatbot_with_mocks):
        """Verify that the chat method sends MODEL_LLM to Ollama"""
        chatbot, mock_client = chatbot_with_mocks
        
        with patch('builtins.print'):
            chatbot.chat("What is Beckhoff?")
        
        # Verify chat was called
        assert mock_client.chat.called
        # Get the actual call arguments
        call_args, call_kwargs = mock_client.chat.call_args
        # MODEL_LLM should be passed
        assert 'model' in call_kwargs
        assert call_kwargs['model'] == 'gemma4:31b-cloud'
    
    def test_ollama_receives_streaming_enabled(self, chatbot_with_mocks):
        """Verify that streaming is enabled in Ollama calls"""
        chatbot, mock_client = chatbot_with_mocks
        
        with patch('builtins.print'):
            chatbot.chat("Test")
        
        call_args, call_kwargs = mock_client.chat.call_args
        assert call_kwargs['stream'] is True
    
    def test_ollama_receives_messages_with_system_prompt(self, chatbot_with_mocks):
        """Verify that system prompt is included in Ollama messages"""
        chatbot, mock_client = chatbot_with_mocks
        
        with patch('builtins.print'):
            chatbot.chat("Test question")
        
        call_args, call_kwargs = mock_client.chat.call_args
        messages = call_kwargs['messages']
        
        # First message should be system prompt
        assert messages[0]['role'] == 'system'
        assert SYSTEM_PROMPT in messages[0]['content']
    
    def test_host_parameter_used_when_local(self):
        """Test that local mode passes HOST_LLM to Ollama Client"""
        with patch('src.llm_chatbot.Rag'):
            with patch('src.llm_chatbot.Client') as mock_client_class:
                LLMChatbot(local=True)
                
                mock_client_class.assert_called_once()
                call_kwargs = mock_client_class.call_args[1]
                assert 'host' in call_kwargs
    
    def test_default_client_used_when_cloud(self):
        """Test that cloud mode uses default ollama.Client when no API key"""
        with patch('src.llm_chatbot.Rag'):
            with patch('src.llm_chatbot.Client') as mock_ollama_client:
                with patch('src.llm_chatbot.os.getenv', return_value=None):
                    LLMChatbot(local=False)
                    
                    mock_ollama_client.assert_called_once_with()
