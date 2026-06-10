import shutil
import os
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config import collection_name, path_to_data

class Rag:
    def __init__(self):
        print("### Creating vectorstore ###")
        
        # delete old database, if exists
        persist_dir = "./chroma_db"
        
        if os.path.exists(persist_dir):
           shutil.rmtree(persist_dir)
        
        # load data - convert html directly to documents
        docs = []
        for html_file in Path(f"{path_to_data}").glob("*.html"):
            html_content = html_file.read_text(encoding="utf-8")
            
            soup = BeautifulSoup(html_content, "html.parser")
            for tag in soup(["button", "nav", "input", "form", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "type": "webpage",
                        "topic": html_file.stem,
                        "filename": html_file.name
                    },
                )
            )
        
        for doc in docs:
            print(f"Loaded document: {doc.metadata['topic']} with {len(doc.page_content)} characters.")
        
        # splitting document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        
        documents = text_splitter.split_documents(docs)
        
        print(f"Number of Chunks: {len(documents)}\n")
        
        if not documents:
            print("No documents found")
            self.db = None
            print("### didn't create vectorstore ###")
            return
            
        # create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="microsoft/harrier-oss-v1-0.6b")
        
        # create vectorstore
        self.db = Chroma(
            collection_name=f"{collection_name}_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        
        self.db.add_documents(documents)
        
        print("### Vectorstore created ###")
    
    def retrieve_data(self, userinput) -> str:
        if not self.db:
            print("Warning: database is empty.")
            return ""
        
        results = self.db.similarity_search(userinput, k=3)
        return "\n".join(doc.page_content for doc in results)
