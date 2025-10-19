import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from tqdm import tqdm

from config.config import config
from oil_rag.retrieval.embedder import DocumentEmbedder
from oil_rag.retrieval.indexer import FAISSIndexer
from oil_rag.utils.logger import setup_logger


def load_documents_from_aligned() -> list:
    logger = setup_logger("load_documents")
    
    aligned_dir = config.paths.aligned_data
    
    if not aligned_dir.exists():
        logger.error(f"Aligned data directory not found: {aligned_dir}")
        return []
    
    documents = []
    
    for jsonl_file in sorted(aligned_dir.glob("aligned_*.jsonl")):
        if "stats" in jsonl_file.name:
            continue
        
        logger.info(f"Loading {jsonl_file.name}")
        
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                pair = json.loads(line)
                
                en_doc = pair["en"]
                no_doc = pair["no"]
                
                en_doc["id"] = f"{pair['pair_id']}_en"
                no_doc["id"] = f"{pair['pair_id']}_no"
                
                documents.append(en_doc)
                documents.append(no_doc)
    
    logger.info(f"Loaded {len(documents)} total documents")
    return documents


def build_faiss_index(documents: list, embedder: DocumentEmbedder, indexer: FAISSIndexer):
    logger = setup_logger("build_index")
    
    logger.info("Generating embeddings...")
    
    batch_size = 32
    all_embeddings = []
    
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i+batch_size]
        texts = [doc["text"] for doc in batch]
        embeddings = embedder.embed_texts(texts, normalize=True, show_progress=False)
        all_embeddings.append(embeddings)
    
    all_embeddings = np.vstack(all_embeddings)
    
    logger.info(f"Generated {all_embeddings.shape[0]} embeddings")
    logger.info(f"Embedding dimension: {all_embeddings.shape[1]}")
    
    logger.info("Adding to FAISS index...")
    indexer.add(all_embeddings, documents)
    
    logger.info(f"Index size: {indexer.get_index_size()}")
    
    return indexer


def main():
    logger = setup_logger("main")
    
    logger.info("Starting index build process")
    logger.info(f"Output index: {config.paths.faiss_index}")
    logger.info(f"Output documents: {config.paths.documents_pkl}")
    
    documents = load_documents_from_aligned()
    
    if not documents:
        logger.error("No documents loaded. Exiting.")
        return
    
    logger.info(f"Building embedder with model: {config.models.embedding_model}")
    embedder = DocumentEmbedder(
        model_name=config.models.embedding_model,
        device=config.models.device
    )
    
    logger.info("Initializing FAISS indexer")
    indexer = FAISSIndexer(
        dimension=config.models.embedding_dim,
        index_type="IVF"
    )
    
    indexer = build_faiss_index(documents, embedder, indexer)
    
    logger.info("Saving index and documents...")
    indexer.save()
    
    logger.info("Index build complete!")
    logger.info(f"Total documents indexed: {indexer.get_index_size()}")


if __name__ == "__main__":
    main()