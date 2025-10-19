import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import argparse
import numpy as np

from oil_rag.retrieval.retriever import BilingualRetriever
from oil_rag.utils.logger import setup_logger


def parse_arguments():
    parser = argparse.ArgumentParser(description="Build retrieval index")
    parser.add_argument(
        "--data_dir",
        default="data/processed",
        help="Directory containing processed data"
    )
    parser.add_argument(
        "--output_dir",
        default="data/index",
        help="Output directory for index"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "no"],
        help="Languages to index"
    )
    return parser.parse_args()


def load_chunks(file_path: str) -> list:
    chunks = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def main():
    args = parse_arguments()
    logger = setup_logger("build_index")
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    retriever = BilingualRetriever()
    
    for language in args.languages:
        logger.info(f"Building index for {language}")
        
        all_chunks = []
        paragraphs_dir = data_dir / "paragraphs"
        
        for chunk_file in paragraphs_dir.glob(f"{language}_*.jsonl"):
            logger.info(f"Loading {chunk_file.name}")
            chunks = load_chunks(str(chunk_file))
            all_chunks.extend(chunks)
        
        logger.info(f"Loaded {len(all_chunks)} chunks for {language}")
        
        logger.info("Adding documents to retriever")
        retriever.add_documents(all_chunks, language)
        
        logger.info(f"Saving embeddings for {language}")
        embeddings = retriever.index[language]
        np.save(output_dir / f"{language}_embeddings.npy", embeddings)
        
        with open(output_dir / f"{language}_chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    
    logger.info("Index building complete")


if __name__ == "__main__":
    main()
