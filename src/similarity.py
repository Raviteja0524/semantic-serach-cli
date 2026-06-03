import numpy as np



def dot_product(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_prod = dot_product(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_prod / (norm_vec1 * norm_vec2)


def l2_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.linalg.norm(vec1 - vec2)

def batch_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    dot_prod = np.dot(query, matrix.T)
    norm_query = np.linalg.norm(query, keepdims=True)
    norm_matrix = np.linalg.norm(matrix, axis=1)

    norm_product = norm_query * norm_matrix
    norm_product[norm_product == 0] = 1e-10  # Avoid division by zero

    return dot_prod / norm_product