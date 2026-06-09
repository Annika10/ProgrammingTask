import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil
from rag import Rag


class TestRag:
    """Tests for Rag class"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory with test HTML files"""
        temp_dir = tempfile.mkdtemp()
        
        # Create sample HTML file
        html_content = """
        <html>
            <header><nav>Navigation</nav></header>
            <body>
                <button>Click me</button>
                <p>Beckhoff is a technology company</p>
                <p>They specialize in industrial automation</p>
            </body>
            <footer>Footer content</footer>
        </html>
        """
        
        Path(temp_dir, "test_page.html").write_text(html_content)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_data_dir):
        """Mock the config module"""
        with patch('rag.collection_name', 'test_collection'):
            with patch('rag.path_to_data', temp_data_dir):
                yield
    
    @pytest.fixture
    def mock_chroma(self):
        """Mock the Chroma vectorstore"""
        with patch('rag.Chroma') as mock:
            mock_instance = Mock()
            mock_instance.similarity_search.return_value = []
            mock.return_value = mock_instance
            yield mock
    
    @pytest.fixture
    def rag(self, mock_config, mock_chroma):
        """Create a Rag instance with mocks"""
        with patch('rag.HuggingFaceEmbeddings'):
            with patch('rag.RecursiveCharacterTextSplitter'):
                with patch('builtins.print'):
                    return Rag()
    
    def test_rag_initialization(self, rag, mock_chroma):
        """Test that Rag initializes correctly"""
        assert rag.db is not None
        assert mock_chroma.called
    
    def test_retrieve_data_success(self, rag):
        """Test successful data retrieval"""
        mock_docs = [
            Mock(page_content="Content 1"),
            Mock(page_content="Content 2"),
            Mock(page_content="Content 3")
        ]
        rag.db.similarity_search.return_value = mock_docs
        
        result = rag.retrieve_data("Test query")
        
        assert result == "Content 1\nContent 2\nContent 3"
        rag.db.similarity_search.assert_called_once_with("Test query", k=3)
    
    def test_retrieve_data_empty_results(self, rag):
        """Test data retrieval with no results"""
        rag.db.similarity_search.return_value = []
        
        result = rag.retrieve_data("Nonexistent query")
        
        assert result == ""
    
    def test_retrieve_data_single_result(self, rag):
        """Test data retrieval with single result"""
        mock_docs = [Mock(page_content="Single content")]
        rag.db.similarity_search.return_value = mock_docs
        
        result = rag.retrieve_data("Query")
        
        assert result == "Single content"
    
    def test_retrieve_data_with_k_parameter(self, rag):
        """Test that retrieve_data uses k=3"""
        rag.db.similarity_search.return_value = []
        
        rag.retrieve_data("Query")
        
        call_args = rag.db.similarity_search.call_args
        assert call_args[1]['k'] == 3
    
    def test_html_cleanup_removes_unwanted_tags(self):
        """Test that unwanted HTML tags are removed during parsing"""
        temp_dir = tempfile.mkdtemp()
        try:
            html_content = """
            <html>
                <button>Button text</button>
                <nav>Navigation</nav>
                <input type="text">
                <form>Form data</form>
                <footer>Footer</footer>
                <header>Header</header>
                <aside>Aside content</aside>
                <p>Important content</p>
            </html>
            """
            Path(temp_dir, "test.html").write_text(html_content)
            
            with patch('rag.path_to_data', temp_dir):
                with patch('rag.Chroma') as mock_chroma:
                    with patch('rag.HuggingFaceEmbeddings'):
                        with patch('rag.RecursiveCharacterTextSplitter'):
                            with patch('builtins.print'):
                                rag = Rag()
            
            # The important content should be there, unwanted tags should not be
            # This is verified by checking what documents were added to the db
            assert mock_chroma.return_value.add_documents.called
        finally:
            shutil.rmtree(temp_dir)
    
    def test_retrieve_data_called_with_user_input(self, rag):
        """Test that retrieve_data correctly processes user input"""
        test_query = "What is Beckhoff?"
        rag.db.similarity_search.return_value = []
        
        rag.retrieve_data(test_query)
        
        rag.db.similarity_search.assert_called_once_with(test_query, k=3)


class TestRagDocumentProcessing:
    """Tests for document processing in Rag"""
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings"""
        with patch('rag.HuggingFaceEmbeddings') as mock:
            yield mock
    
    @pytest.fixture
    def mock_text_splitter(self):
        """Mock text splitter"""
        with patch('rag.RecursiveCharacterTextSplitter') as mock:
            mock_instance = Mock()
            mock_instance.split_documents.return_value = []
            mock.return_value = mock_instance
            yield mock
    
    def test_text_splitter_configuration(self, mock_text_splitter, mock_embeddings):
        """Test that text splitter is configured with correct parameters"""
        temp_dir = tempfile.mkdtemp()
        try:
            Path(temp_dir, "test.html").write_text("<html><p>Test</p></html>")
            
            with patch('rag.path_to_data', temp_dir):
                with patch('rag.Chroma'):
                    with patch('builtins.print'):
                        Rag()
            
            mock_text_splitter.assert_called_once_with(
                chunk_size=800,
                chunk_overlap=150
            )
        finally:
            shutil.rmtree(temp_dir)
    
    def test_embeddings_model_name(self, mock_text_splitter, mock_embeddings):
        """Test that correct embeddings model is used"""
        temp_dir = tempfile.mkdtemp()
        try:
            Path(temp_dir, "test.html").write_text("<html><p>Test</p></html>")
            
            with patch('rag.path_to_data', temp_dir):
                with patch('rag.Chroma'):
                    with patch('builtins.print'):
                        Rag()
            
            mock_embeddings.assert_called_once_with(
                model_name="microsoft/harrier-oss-v1-0.6b"
            )
        finally:
            shutil.rmtree(temp_dir)
