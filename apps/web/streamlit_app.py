import streamlit as st
import sys
from pathlib import Path
import time

project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from config.config import config
from oil_rag.core.system_builder import get_retriever


st.set_page_config(
    page_title="Oil Company RAG System",
    page_icon="🛢️",
    layout="wide"
)


@st.cache_resource
def load_retriever(mode: str, use_reranker: bool):
    return get_retriever(mode=mode, use_reranker=use_reranker)


def main():
    st.title("🛢️ Oil Company Reports RAG System")
    st.markdown("Hybrid BM25 + Dense + Reranker Retrieval")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        retrieval_mode = st.selectbox(
            "Retrieval Mode",
            ["hybrid", "dense", "bm25"],
            help="hybrid: BM25+Dense fusion, dense: Dense only, bm25: BM25 only"
        )
        
        use_reranker = st.checkbox("Enable Reranker", value=True)
        
        num_results = st.slider("Number of results", 3, 20, 5)
        
        show_scores = st.checkbox("Show scores", value=True)
        show_metadata = st.checkbox("Show metadata", value=True)
        
        st.divider()
        
        st.subheader("About")
        st.write("Searches 18,814 oil company documents")
        st.write(f"Mode: {retrieval_mode.upper()}")
        st.write(f"Reranker: {'✅' if use_reranker else '❌'}")
    
    retriever = load_retriever(retrieval_mode, use_reranker)
    
    sample_queries = [
        "What were safety measures in 2020?",
        "Oil production statistics",
        "Environmental initiatives",
        "Technology development",
        "Financial performance highlights"
    ]
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Sample Queries")
        for i, sample in enumerate(sample_queries):
            if st.button(sample, key=f"sample_{i}"):
                st.session_state.query = sample
    
    with col1:
        query = st.text_input(
            "Enter your question:",
            value=st.session_state.get("query", ""),
            placeholder="What was oil production in 2020?"
        )
        
        if st.button("🔍 Search", type="primary") or query:
            if query:
                with st.spinner("Searching..."):
                    start_time = time.time()
                    
                    documents, scores = retriever.retrieve(
                        query, 
                        k=num_results,
                        use_bm25=(retrieval_mode in ["hybrid", "bm25"]),
                        use_reranker=use_reranker
                    )
                    
                    search_time = time.time() - start_time
                
                st.success(f"Found {len(documents)} results in {search_time:.3f}s")
                
                for i, (doc, score) in enumerate(zip(documents, scores)):
                    with st.expander(f"📄 Document {i+1} - Score: {score:.3f}"):
                        
                        if show_metadata:
                            col_meta1, col_meta2, col_meta3 = st.columns(3)
                            with col_meta1:
                                st.write(f"**Year:** {doc.get('year', 'N/A')}")
                            with col_meta2:
                                st.write(f"**Language:** {doc.get('lang', 'N/A')}")
                            with col_meta3:
                                st.write(f"**Section:** {doc.get('section', 'N/A')[:30]}")
                        
                        st.write("**Content:**")
                        content = doc.get("text", "")
                        if len(content) > 500:
                            content = content[:500] + "..."
                        st.write(content)
                        
                        if show_scores:
                            score_cols = st.columns(3)
                            with score_cols[0]:
                                bm25 = doc.get("bm25_score", "N/A")
                                st.metric("BM25", f"{bm25:.3f}" if isinstance(bm25, float) else bm25)
                            with score_cols[1]:
                                rerank = doc.get("rerank_score", "N/A")
                                st.metric("Rerank", f"{rerank:.3f}" if isinstance(rerank, float) else rerank)
                            with score_cols[2]:
                                st.metric("Final", f"{score:.3f}")
            else:
                st.info("Please enter a query")


if __name__ == "__main__":
    main()