import joblib
import numpy as np

from bert_encoder import get_bert_embedding
from text_preprocessing import clean_webpage_text
from hybrid_fusion import fuse_features
from shap_explainer import explain_prediction

# Load trained model
model = joblib.load("models/hybrid_model.pkl")


def hybrid_predict(feature_dict, html):
    # Engineered features
    engineered = list(feature_dict.values())

    # Clean webpage text
    text = clean_webpage_text(html)

    # DistilBERT embedding
    bert = get_bert_embedding(text)

    # 791-dimensional fused vector
    fused = fuse_features(engineered, bert)

    # Prediction
    probability = model.predict_proba(
        np.array(fused).reshape(1, -1)
    )[0][1]

    prediction = "Phishing" if probability >= 0.5 else "Benign"

    # SHAP explanation
    top_features = explain_prediction(fused)

    return prediction, float(probability), top_features