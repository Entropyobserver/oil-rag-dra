import argparse
import torch
from pathlib import Path

from oil_rag.retrieval.embedder import DocumentEmbedder
from oil_rag.retrieval.indexer import FAISSIndexer
from oil_rag.retrieval.reranker import CrossEncoderReranker
from oil_rag.retrieval.retriever import HybridRetriever
from oil_rag.models.dynamic_lora import DynamicLoRAModel
from oil_rag.models.dra_controller import DRAController, FeatureExtractor
from oil_rag.core.rag_pipeline import DRARAGPipeline
from oil_rag.evaluation.evaluate import RAGEvaluator, AblationStudy
from oil_rag.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG experiments")
    parser.add_argument("--mode", type=str, required=True, choices=['evaluate', 'ablation', 'both'])
    parser.add_argument("--test_data", type=str, default="data/test/test_qa.jsonl")
    parser.add_argument("--index_path", type=str, default="models/faiss_index.bin")
    parser.add_argument("--documents_path", type=str, default="models/documents.pkl")
    parser.add_argument("--model_path", type=str, default="models/dra_model")
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--baseline_ranks", type=int, nargs='+', default=[4, 8, 16, 32])
    return parser.parse_args()


def build_retriever(index_path: str, documents_path: str, device: str) -> HybridRetriever:
    logger = setup_logger('build_retriever')
    logger.info("Building retriever")
    
    embedder = DocumentEmbedder(device=device)
    
    indexer = FAISSIndexer(dimension=768, index_type="IVF")
    indexer.load(index_path, documents_path)
    
    reranker = CrossEncoderReranker(device=device)
    
    retriever = HybridRetriever(
        embedder=embedder,
        indexer=indexer,
        reranker=reranker,
        initial_k=100,
        final_k=10
    )
    
    logger.info(f"Retriever built with {indexer.get_index_size()} documents")
    return retriever


def build_pipeline(
    retriever: HybridRetriever,
    model_path: str,
    device: str
) -> DRARAGPipeline:
    logger = setup_logger('build_pipeline')
    logger.info("Building DRA pipeline")
    
    generator = DynamicLoRAModel(
        base_model_name="google/mt5-base",
        r_max=32
    )
    
    if Path(model_path).exists():
        generator.load_pretrained(model_path)
        logger.info(f"Loaded generator from {model_path}")
    
    dra_controller = DRAController(
        input_dim=8,
        hidden_dims=[32, 16],
        r_min=4,
        r_max=32
    )
    
    checkpoint_path = Path(model_path) / "dra_controller.pt"
    if checkpoint_path.exists():
        dra_controller.load_state_dict(torch.load(checkpoint_path))
        logger.info("Loaded DRA controller checkpoint")
    
    feature_extractor = FeatureExtractor()
    
    pipeline = DRARAGPipeline(
        retriever=retriever,
        generator=generator,
        dra_controller=dra_controller,
        feature_extractor=feature_extractor,
        device=device
    )
    
    logger.info("Pipeline built successfully")
    return pipeline


def run_evaluation(args):
    logger = setup_logger('run_evaluation')
    logger.info("Starting evaluation")
    
    retriever = build_retriever(args.index_path, args.documents_path, args.device)
    pipeline = build_pipeline(retriever, args.model_path, args.device)
    
    evaluator = RAGEvaluator(
        test_data_path=args.test_data,
        output_dir=args.output_dir
    )
    
    results = evaluator.evaluate_baselines(
        dra_pipeline=pipeline,
        baseline_ranks=args.baseline_ranks
    )
    
    logger.info("Evaluation completed")
    
    complexity_analysis = evaluator.analyze_query_complexity()
    logger.info(f"Query complexity analysis: {complexity_analysis}")
    
    return results


def run_ablation(args):
    logger = setup_logger('run_ablation')
    logger.info("Starting ablation study")
    
    retriever = build_retriever(args.index_path, args.documents_path, args.device)
    pipeline = build_pipeline(retriever, args.model_path, args.device)
    
    evaluator = RAGEvaluator(test_data_path=args.test_data)
    test_data = evaluator.test_data
    
    ablation = AblationStudy(pipeline, test_data)
    results = ablation.run_feature_ablation()
    
    logger.info("Ablation study completed")
    return results


def main():
    args = parse_args()
    logger = setup_logger('main')
    
    logger.info(f"Running experiments in {args.mode} mode")
    logger.info(f"Test data: {args.test_data}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.mode in ['evaluate', 'both']:
        eval_results = run_evaluation(args)
        logger.info(f"Evaluation results: {eval_results}")
    
    if args.mode in ['ablation', 'both']:
        ablation_results = run_ablation(args)
        logger.info(f"Ablation study completed: {len(ablation_results)} configurations tested")
    
    logger.info("All experiments completed successfully")


if __name__ == "__main__":
    main()