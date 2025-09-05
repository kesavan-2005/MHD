import numpy as np
from sklearn.ensemble import RandomForestClassifier

class CombinedModel:
    def __init__(self):
        self.meta_model = RandomForestClassifier()

    def fit(self, cnn_preds, lstm_preds, text_preds, labels):
        features = np.concatenate([cnn_preds, lstm_preds, text_preds], axis=1)
        self.meta_model.fit(features, labels)

    def predict(self, cnn_pred, lstm_pred, text_pred):
        features = np.concatenate([cnn_pred, lstm_pred, text_pred], axis=1)
        return self.meta_model.predict(features)
