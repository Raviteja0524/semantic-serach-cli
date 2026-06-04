from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from sentence_transformers import SentenceTransformer


@dataclass
class EmbedderConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    max_retries: int = 3
    retry_delay: float = 1.0
    provider: str = 'local'  


def retry_with_backoff(max_retries: int, delay: float) -> Callable:
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Error: {e}. Retrying in {delay} seconds...")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        print(f"Error: {e}. Max retries reached. Giving up.")
                        raise
        return wrapper
    return decorator

class Embedder:
    def __init__(self, config: EmbedderConfig):
        self.config = config
        self.model = self._load_model()

    @retry_with_backoff(max_retries=3, delay=1.0)
    def _load_model(self):
        if self.config.provider == 'local':
            return SentenceTransformer(self.config.model_name)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    @retry_with_backoff(max_retries=3, delay=1.0)
    def embed(self, text: str) -> np.ndarray:
        return np.array(self.model.encode(text))
        
    @retry_with_backoff(max_retries=3, delay=1.0)
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, batch_size=self.config.batch_size))

