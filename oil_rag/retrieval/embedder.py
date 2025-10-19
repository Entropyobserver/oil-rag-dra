from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class DocumentEmbedder:
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        max_length: int = 512
    ):
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    def embed_texts(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            device=self.device
        )
        
        return embeddings

    def embed_documents(
        self,
        documents: List[dict],
        text_field: str = "text"
    ) -> np.ndarray:
        texts = [doc[text_field] for doc in documents]
        return self.embed_texts(texts, show_progress=True)