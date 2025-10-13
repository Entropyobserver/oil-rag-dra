from typing import List, Dict
import sys
sys.path.append('.')

import json
import csv
import argparse
from pathlib import Path
import numpy as np

from src.data.pdf_parser import PDFParser
from src.data.enhanced_chunker import EnhancedChunker
from src.data.aligner import BilingualAligner
from src.utils.logger import setup_logger


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, help='Process specific year only')
    parser.add_argument('--pairs_file', default='data/metadata/report_pairs.csv')
    parser.add_argument('--reports_dir', default='data/reports')
    parser.add_argument('--output_dir', default='data/processed')
    return parser.parse_args()


def load_report_pairs(pairs_file: str):
    pairs = []
    with open(pairs_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(row)
    return pairs


def process_report(pdf_path: str, year: int, lang: str, chunker: EnhancedChunker):
    parser = PDFParser()
    pages = parser.parse_pdf(pdf_path)
    chunks = chunker.chunk_with_metadata(pages, year, lang)
    return chunks


def save_chunks(chunks: List, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            # Convert numpy types to Python native types
            chunk_converted = convert_numpy_types(chunk)
            f.write(json.dumps(chunk_converted, ensure_ascii=False) + '\n')


def save_alignments(alignments: List, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for alignment in alignments:
            # Convert numpy types to Python native types
            alignment_converted = convert_numpy_types(alignment)
            f.write(json.dumps(alignment_converted, ensure_ascii=False) + '\n')


def compute_statistics(alignments: List, en_chunks: List, no_chunks: List):
    total_en = len(en_chunks)
    total_no = len(no_chunks)
    total_aligned = len(alignments)
    
    scores = [a['alignment']['score'] for a in alignments]
    avg_score = np.mean(scores) if scores else 0.0
    
    high_conf = sum(1 for a in alignments if a['alignment']['confidence'] == 'high')
    medium_conf = sum(1 for a in alignments if a['alignment']['confidence'] == 'medium')
    low_conf = sum(1 for a in alignments if a['alignment']['confidence'] == 'low')
    
    bidirectional = sum(1 for a in alignments if a['alignment']['bidirectional_match'])
    
    return {
        'total_en_chunks': total_en,
        'total_no_chunks': total_no,
        'total_alignments': total_aligned,
        'coverage_en': total_aligned / total_en if total_en > 0 else 0,
        'coverage_no': total_aligned / total_no if total_no > 0 else 0,
        'avg_similarity': float(avg_score),
        'high_confidence': high_conf,
        'medium_confidence': medium_conf,
        'low_confidence': low_conf,
        'bidirectional_matches': bidirectional,
        'bidirectional_ratio': bidirectional / total_aligned if total_aligned > 0 else 0
    }


def main():
    args = get_args()
    logger = setup_logger('align_reports')
    
    reports_dir = Path(args.reports_dir)
    pairs_file = args.pairs_file
    output_dir = Path(args.output_dir)
    
    pairs = load_report_pairs(pairs_file)
    
    if args.year:
        pairs = [p for p in pairs if int(p['year']) == args.year]
        logger.info(f"Processing only year {args.year}")
    
    logger.info(f"Loaded {len(pairs)} report pairs")
    
    chunker = EnhancedChunker(min_words=30, max_words=150)
    aligner = BilingualAligner(threshold=0.70)
    
    for pair in pairs:
        year = int(pair['year'])
        en_report = pair['en_report']
        no_report = pair['no_report']
        
        logger.info(f"Processing year {year}")
        
        en_path = reports_dir / 'en' / en_report
        no_path = reports_dir / 'no' / no_report
        
        if not en_path.exists() or not no_path.exists():
            logger.warning(f"Missing reports for year {year}, skipping")
            continue
        
        logger.info(f"Extracting English chunks from {en_report}")
        en_chunks = process_report(str(en_path), year, 'en', chunker)
        logger.info(f"Extracted {len(en_chunks)} English chunks")
        
        logger.info(f"Extracting Norwegian chunks from {no_report}")
        no_chunks = process_report(str(no_path), year, 'no', chunker)
        logger.info(f"Extracted {len(no_chunks)} Norwegian chunks")
        
        save_chunks(en_chunks, output_dir / 'paragraphs' / f'en_{year}.jsonl')
        save_chunks(no_chunks, output_dir / 'paragraphs' / f'no_{year}.jsonl')
        
        logger.info("Encoding chunks")
        en_embeddings = aligner.encode_chunks(en_chunks)
        no_embeddings = aligner.encode_chunks(no_chunks)
        
        np.save(output_dir / 'embeddings' / f'en_{year}.npy', en_embeddings)
        np.save(output_dir / 'embeddings' / f'no_{year}.npy', no_embeddings)
        
        logger.info("Aligning chunks")
        alignments = aligner.align_chunks(en_chunks, no_chunks, en_embeddings, no_embeddings)
        logger.info(f"Created {len(alignments)} alignments")
        
        save_alignments(alignments, output_dir / 'aligned' / f'aligned_{year}.jsonl')
        
        stats = compute_statistics(alignments, en_chunks, no_chunks)
        logger.info(f"Statistics: {stats}")
        
        stats_file = output_dir / 'aligned' / f'aligned_{year}_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    logger.info("Alignment complete for all reports")


if __name__ == '__main__':
    main()