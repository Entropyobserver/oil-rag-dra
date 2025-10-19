import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import random
from typing import List, Dict

from oil_rag.utils.logger import setup_logger


def load_alignments(file_path: str) -> List[Dict]:
    alignments = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            alignments.append(json.loads(line))
    return alignments


def sample_alignments(alignments: List[Dict], n: int = 100) -> List[Dict]:
    if len(alignments) <= n:
        return alignments
    return random.sample(alignments, n)


def display_alignment(alignment: Dict, index: int):
    separator = "=" * 80
    print(f"\n{separator}")
    print(f"Sample {index + 1}")
    print(separator)
    print(f"Pair ID: {alignment['pair_id']}")
    print(f"Score: {alignment['alignment']['score']:.4f}")
    print(f"Confidence: {alignment['alignment']['confidence']}")
    print(f"Bidirectional: {alignment['alignment']['bidirectional_match']}")
    print(f"\nEnglish ({alignment['en']['word_count']} words):")
    print(f"Section: {alignment['en']['section']}")
    print(f"Text: {alignment['en']['text'][:300]}...")
    print(f"\nNorwegian ({alignment['no']['word_count']} words):")
    print(f"Section: {alignment['no']['section']}")
    print(f"Text: {alignment['no']['text'][:300]}...")


def manual_validation(alignments: List[Dict]) -> List[Dict]:
    separator = "=" * 80
    print(f"\n{separator}")
    print("MANUAL VALIDATION")
    print(separator)
    print("For each pair, rate: 1=Wrong, 2=Partial, 3=Correct")
    print("Press Enter to skip")
    
    ratings = []
    
    for i, alignment in enumerate(alignments):
        display_alignment(alignment, i)
        
        while True:
            try:
                rating = input("\nRating (1/2/3 or Enter to skip): ").strip()
                if rating == "":
                    break
                rating = int(rating)
                if rating in [1, 2, 3]:
                    ratings.append({
                        "pair_id": alignment["pair_id"],
                        "rating": rating,
                        "score": alignment["alignment"]["score"],
                        "confidence": alignment["alignment"]["confidence"]
                    })
                    break
                else:
                    print("Please enter 1, 2, or 3")
            except ValueError:
                print("Invalid input")
    
    return ratings


def compute_validation_metrics(ratings: List[Dict]) -> Dict:
    if not ratings:
        return None
    
    correct = sum(1 for r in ratings if r["rating"] == 3)
    partial = sum(1 for r in ratings if r["rating"] == 2)
    wrong = sum(1 for r in ratings if r["rating"] == 1)
    total = len(ratings)
    
    precision = correct / total if total > 0 else 0
    
    by_confidence = {}
    for conf in ["high", "medium", "low"]:
        conf_ratings = [r for r in ratings if r["confidence"] == conf]
        if conf_ratings:
            conf_correct = sum(1 for r in conf_ratings if r["rating"] == 3)
            by_confidence[conf] = {
                "total": len(conf_ratings),
                "correct": conf_correct,
                "precision": conf_correct / len(conf_ratings)
            }
    
    return {
        "total_validated": total,
        "correct": correct,
        "partial": partial,
        "wrong": wrong,
        "precision": precision,
        "by_confidence": by_confidence
    }


def analyze_errors(ratings: List[Dict]):
    errors = [r for r in ratings if r["rating"] == 1]
    
    if not errors:
        print("\nNo errors found!")
        return
    
    separator = "=" * 80
    print(f"\n{separator}")
    print(f"ERROR ANALYSIS ({len(errors)} errors)")
    print(separator)
    
    avg_error_score = sum(e["score"] for e in errors) / len(errors)
    print(f"Average score of errors: {avg_error_score:.4f}")
    
    error_by_conf = {}
    for e in errors:
        conf = e["confidence"]
        error_by_conf[conf] = error_by_conf.get(conf, 0) + 1
    
    print("\nErrors by confidence:")
    for conf, count in error_by_conf.items():
        print(f"  {conf}: {count}")


def main():
    logger = setup_logger("validate_alignment")
    
    aligned_dir = Path("data/processed/aligned")
    
    if not aligned_dir.exists():
        logger.error(f"Directory not found: {aligned_dir}")
        return
    
    all_alignments = []
    for aligned_file in aligned_dir.glob("aligned_*.jsonl"):
        if "stats" not in aligned_file.name:
            logger.info(f"Loading {aligned_file.name}")
            alignments = load_alignments(str(aligned_file))
            all_alignments.extend(alignments)
    
    logger.info(f"Loaded {len(all_alignments)} total alignments")
    
    sample_size = min(100, len(all_alignments))
    sampled = sample_alignments(all_alignments, sample_size)
    logger.info(f"Sampled {len(sampled)} alignments for validation")
    
    ratings = manual_validation(sampled)
    
    if ratings:
        logger.info(f"Collected {len(ratings)} ratings")
        
        metrics = compute_validation_metrics(ratings)
        
        separator = "=" * 80
        print(f"\n{separator}")
        print("VALIDATION METRICS")
        print(separator)
        print(f"Total validated: {metrics['total_validated']}")
        print(f"Correct: {metrics['correct']}")
        print(f"Partial: {metrics['partial']}")
        print(f"Wrong: {metrics['wrong']}")
        print(f"Precision: {metrics['precision']:.4f}")
        
        print("\nBy confidence level:")
        for conf, stats in metrics["by_confidence"].items():
            print(
                f"  {conf}: {stats['correct']}/{stats['total']} "
                f"(precision: {stats['precision']:.4f})"
            )
        
        analyze_errors(ratings)
        
        output_file = aligned_dir / "validation_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {"metrics": metrics, "ratings": ratings},
                f,
                indent=2,
                ensure_ascii=False
            )
        
        logger.info(f"Saved validation results to {output_file}")
    else:
        logger.info("No ratings collected")


if __name__ == "__main__":
    main()
