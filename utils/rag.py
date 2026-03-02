from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class RAG:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.PERSIST_DIRECTORY = "./chroma_rag_db"

        self._build_vector_store()
    
    def _load_documents(self) -> str:
        loader = DirectoryLoader(
            self.directory_path, 
            glob="**/*.txt", 
            loader_cls=TextLoader
        )
        return loader.load()
    
    def _chunk_documents(self):
        documents = self._load_documents()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        return chunks

    def _build_vector_store(self):
        chunks = self._chunk_documents()
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.PERSIST_DIRECTORY
        )
        return self.vector_store
    
    def retrieve(self, query: str, top_k: int = 3):
        results = self.vector_store.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]
    
