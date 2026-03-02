import uuid
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
PERSIST_DIRECTORY = "./chroma_memory_db"

memory_store = Chroma(
    collection_name="solved_problems",
    embedding_function=embeddings_model,
    persist_directory=PERSIST_DIRECTORY
)

def save_memory(original_input: str, retrieved_context: str, 
                final_answer: str, verifier_outcome: str, human_feedback: str):
    metadata = {
        "retrieved_context": str(retrieved_context) if retrieved_context else "",
        "final_answer": str(final_answer) if final_answer else "",
        "verifier_outcome": str(verifier_outcome) if verifier_outcome else "Approved",
        "human_feedback": str(human_feedback) if human_feedback else "approve"
    }
    
    memory_store.add_texts(
        texts=[original_input],
        metadatas=[metadata],
        ids=[str(uuid.uuid4())]
    )

def search_memory(query: str, limit: int = 3, threshold: float = 0.7) -> list[dict]:
    results = memory_store.similarity_search_with_score(query, k=limit)
    
    formatted_results = []
    for doc, similarity in results:

        score = 1 - similarity
        if score >= threshold:
            formatted_results.append({
                "original_input": doc.page_content, 
                "final_answer": doc.metadata.get("final_answer", ""), 
                "score": score
            })

    return formatted_results