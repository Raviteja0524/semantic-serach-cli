from dataclasses import dataclass
import numpy as np

@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    rank: int



class VectorStore:
    def __init__(self):
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.embeddings: np.ndarray = np.empty((0, 0))

    def add(self, id: str, text: str, embedding: np.ndarray) -> None:
        self.ids.append(id)
        self.texts.append(text)
        self.embeddings = np.vstack([self.embeddings, embedding]) if self.embeddings.size else embedding

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        if self.embeddings.size == 0:
            return []

        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        top_k_indices = similarities.argsort()[-top_k:][::-1]
        results = [
            SearchResult(id=self.ids[i], text=self.texts[i], score=similarities[i], rank=i + 1)
            for i in top_k_indices
        ]
        return results

    def delete(self, id: str) -> None:
        if id in self.ids:
            index = self.ids.index(id)
            self.ids.pop(index)
            self.texts.pop(index)
            self.embeddings = np.delete(self.embeddings, index, axis=0)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            np.savez(f, ids=self.ids, texts=self.texts, embeddings=self.embeddings)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = np.load(f)
            self.ids = data['ids'].tolist()
            self.texts = data['texts'].tolist()
            self.embeddings = data['embeddings']