import numpy as np

def fuse_features(url_html_features, bert_embedding):
    """
    Combines:
    - 23 engineered URL/HTML features
    - 768-dimensional BERT embedding

    Returns one fused feature vector.
    """

    engineered = np.array(url_html_features, dtype=np.float32)
    bert = np.array(bert_embedding, dtype=np.float32)

    fused = np.concatenate([engineered, bert])

    return fused