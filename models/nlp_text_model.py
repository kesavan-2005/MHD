from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def build_text_model():
    model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=200))
    return model
