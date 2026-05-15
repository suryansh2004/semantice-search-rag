import numpy as np


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(
                "Could not load embedding model "
                f"'{model_name}'. Set RAG_EMBEDDING_MODEL to a local model path "
                "or allow the model to be downloaded before starting the service."
            ) from exc

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)
