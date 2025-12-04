import numpy as np
from functools import lru_cache
from .helper import decode_image, encode_image

@lru_cache(maxsize=32)
def compute_svd_matrix(contents):
    """Tính toán SVD một lần và cache kết quả."""
    img = decode_image(contents)
    if img is None: return None
    
    # SVD: Phân rã ma trận A = U * S * Vt
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    return U, s, Vt, img.shape

def process_svd(contents, k):
    """
    Tái tạo ảnh dùng SVD với k thành phần.
    """
    data = compute_svd_matrix(contents)
    if data is None: return None, 0, [], "0 KB"

    U, s, Vt, shape = data
    
    # Đảm bảo k không vượt quá kích thước thật
    k = min(k, len(s))

    # 1. Tái tạo ảnh: A_k = U_k * S_k * Vt_k [cite: 19]
    S_k = np.diag(s[:k])
    reconstructed = np.dot(U[:, :k], np.dot(S_k, Vt[:k, :]))
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    # 2. Tính phương sai (Mức độ thông tin giữ lại)
    eigenvalues = s ** 2
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    retention = cumulative_variance[k-1] * 100
    
    # 3. Ước lượng dung lượng nén (Lý thuyết)
    compressed_size = (U.shape[0]*k + k + k*Vt.shape[1]) * 4 # giả sử 4 byte/float
    size_str = f"{compressed_size/1024:.2f} KB"

    return encode_image(reconstructed), retention, cumulative_variance, size_str