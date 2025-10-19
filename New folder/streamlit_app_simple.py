import streamlit as st
import sys
from pathlib import Path
import time

project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from oil_rag.retrieval.embedder import DocumentEmbedder
from oil_rag.retrieval.indexer import FAISSIndexer
from oil_rag.retrieval.retriever import HybridRetriever

@st.cache_resource
def load_rag_system():
    embedder = DocumentEmbedder(device="cpu")
    indexer = FAISSIndexer(dimension=768, index_type="IVF")
    indexer.load("models/faiss_index.bin", "models/documents.pkl")
    
    retriever = HybridRetriever(
        embedder=embedder,
        indexer=indexer,
        reranker=None,
        initial_k=50,
        final_k=5
    )
    return retriever

def generate_answer(retrieved_docs):
    if not retrieved_docs:
        return "No relevant information found"
    
    combined_text = ""
    for doc in retrieved_docs[:3]:
        text = doc.get('text', '')
        sentences = text.split('.')[:2]
        combined_text += '. '.join(sentences) + ". "
    
    if len(combined_text) > 500:
        combined_text = combined_text[:500] + "..."
    
    return combined_text.strip()

def search_documents(retriever, query, num_results):
    start_time = time.time()
    docs, scores = retriever.retrieve(query, k=num_results)
    search_time = time.time() - start_time
    return docs, scores, search_time

def main():
    st.set_page_config(
        page_title="Oil Company RAG System",
        page_icon="🛢️",
        layout="wide"
    )
    
    st.title("Oil Company Reports RAG System")
    st.markdown("Search and ask questions about oil company annual reports")
    
    with st.spinner("Loading RAG system..."):
        retriever = load_rag_system()
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Settings")
        num_results = st.slider("Number of results", 3, 10, 5)
        show_metadata = st.checkbox("Show document metadata", True)
        show_scores = st.checkbox("Show relevance scores", True)
    
    with col1:
        st.subheader("Ask a Question")
        
        sample_questions = [
            "What were the safety measures implemented in 2020?",
            "How did oil production change over the years?",
            "What environmental initiatives were taken?",
            "What new technologies were developed?",
            "What were the financial performance highlights?"
        ]
        
        selected_sample = st.selectbox("Choose a sample question:", [""] + sample_questions)
        
        query = st.text_input(
            "Enter your question:",
            value=selected_sample,
            placeholder="Type your question about oil company operations..."
        )
        
        if st.button("Search", type="primary") and query:
            with st.spinner("Searching documents..."):
                docs, scores, search_time = search_documents(retriever, query, num_results)
            
            if docs:
                answer = generate_answer(docs)
                
                st.subheader("Generated Answer")
                st.write(answer)
                
                st.subheader("Retrieved Documents")
                
                for i, (doc, score) in enumerate(zip(docs, scores)):
                    with st.expander(f"Document {i+1} - Relevance: {score:.3f}"):
                        
                        if show_metadata:
                            col_meta1, col_meta2, col_meta3 = st.columns(3)
                            with col_meta1:
                                st.write(f"**Year:** {doc.get('year', 'N/A')}")
                            with col_meta2:
                                st.write(f"**Language:** {doc.get('lang', 'N/A')}")
                            with col_meta3:
                                st.write(f"**Section:** {doc.get('section', 'N/A')}")
                        
                        st.write("**Content:**")
                        content = doc.get('text', 'No content available')
                        if len(content) > 800:
                            content = content[:800] + "..."
                        st.write(content)
                        
                        if show_scores:
                            st.write(f"**Relevance Score:** {score:.4f}")
                
                st.sidebar.subheader("Search Statistics")
                st.sidebar.write(f"Search time: {search_time:.3f} seconds")
                st.sidebar.write(f"Documents found: {len(docs)}")
                if scores:
                    st.sidebar.write(f"Average relevance: {sum(scores)/len(scores):.3f}")
            
            else:
                st.warning("No relevant documents found for your query")
        
        elif query and not st.button("Search"):
            st.info("Click Search button to find documents")
    
    with st.sidebar:
        st.subheader("About")
        st.write("This RAG system searches through oil company annual reports from 2010-2024")
        
        st.subheader("Sample Topics")
        st.write("- Safety and environmental measures")
        st.write("- Production volumes and statistics")  
        st.write("- Financial performance")
        st.write("- Technology and innovation")
        st.write("- Strategic initiatives")

if __name__ == "__main__":
    main()