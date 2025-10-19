import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import csv
import argparse
import numpy as np

from oil_rag.data.processors.pdf_parser import PDFParser
from oil_rag.data.processors.chunker import EnhancedChunker
from oil_rag.data.processors.section_extractor import SectionExtractor
from oil_rag.data.processors.aligner import BilingualAligner
from oil_rag.evaluation.metrics import AlignmentMetrics
from oil_rag.utils.logger import setup_logger


def convert_numpy_types(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def parse_arguments():
    parser = argparse.ArgumentParser(description="Align bilingual reports")
    parser.add_argument("--year", type=int, help="Process specific year only")
    parser.add_argument(
        "--pairs_file",
        default="data/metadata/report_pairs.csv",
        help="CSV file containing report pairs"
    )
    parser.add_argument(
        "--reports_dir",
        default="data/reports",
        help="Directory containing PDF reports"
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Similarity threshold for alignment"
    )
    return parser.parse_args()


def load_report_pairs(pairs_file: str) -> list:
    pairs = []
    with open(pairs_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(row)
    return pairs


def process_report(
    pdf_path: str,
    year: int,
    language: str,
    chunker: EnhancedChunker,
    section_extractor: SectionExtractor
) -> list:
    parser = PDFParser()
    pages = parser.parse_pdf(pdf_path)
    
    all_chunks = []
    for page in pages:
        text = page["text"]
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        sections = section_extractor.extract_sections(paragraphs)
        
        for section in sections:
            section_title = section["title"]
            for para in section["paragraphs"]:
                chunks = chunker.chunk_with_metadata(
                    text=para,
                    year=year,
                    language=language,
                    section=section_title,
                    page=page["page"]
                )
                all_chunks.extend(chunks)
    
    return all_chunks


def save_jsonl(data: list, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            converted = convert_numpy_types(item)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")


def save_json(data: dict, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    args = parse_arguments()
    logger = setup_logger("align_reports")
    
    reports_dir = Path(args.reports_dir)
    output_dir = Path(args.output_dir)
    
    pairs = load_report_pairs(args.pairs_file)
    
    if args.year:
        pairs = [p for p in pairs if int(p["year"]) == args.year]
        logger.info(f"Processing only year {args.year}")
    
    logger.info(f"Loaded {len(pairs)} report pairs")
    
    chunker = EnhancedChunker(min_words=30, max_words=150)
    section_extractor = SectionExtractor()
    aligner = BilingualAligner(threshold=args.threshold)
    
    for pair in pairs:
        year = int(pair["year"])
        en_report = pair["en_report"]
        no_report = pair["no_report"]
        
        logger.info(f"Processing year {year}")
        
        en_path = reports_dir / "en" / en_report
        no_path = reports_dir / "no" / no_report
        
        if not en_path.exists() or not no_path.exists():
            logger.warning(f"Missing reports for year {year}, skipping")
            continue
        
        logger.info(f"Extracting English chunks from {en_report}")
        en_chunks = process_report(
            str(en_path), year, "en", chunker, section_extractor
        )
        logger.info(f"Extracted {len(en_chunks)} English chunks")
        
        logger.info(f"Extracting Norwegian chunks from {no_report}")
        no_chunks = process_report(
            str(no_path), year, "no", chunker, section_extractor
        )
        logger.info(f"Extracted {len(no_chunks)} Norwegian chunks")
        
        save_jsonl(en_chunks, output_dir / "paragraphs" / f"en_{year}.jsonl")
        save_jsonl(no_chunks, output_dir / "paragraphs" / f"no_{year}.jsonl")
        
        logger.info("Encoding chunks")
        en_embeddings = aligner.encode_chunks(en_chunks)
        no_embeddings = aligner.encode_chunks(no_chunks)
        
        np.save(output_dir / "embeddings" / f"en_{year}.npy", en_embeddings)
        np.save(output_dir / "embeddings" / f"no_{year}.npy", no_embeddings)
        
        logger.info("Aligning chunks")
        alignments = aligner.align_chunks(
            en_chunks, no_chunks, en_embeddings, no_embeddings
        )
        logger.info(f"Created {len(alignments)} alignments")
        
        save_jsonl(alignments, output_dir / "aligned" / f"aligned_{year}.jsonl")
        
        stats = AlignmentMetrics.compute_statistics(
            alignments, en_chunks, no_chunks
        )
        logger.info(f"Statistics: {stats}")
        
        save_json(stats, output_dir / "aligned" / f"aligned_{year}_stats.json")
    
    logger.info("Alignment complete for all reports")


if __name__ == "__main__":
    main()
