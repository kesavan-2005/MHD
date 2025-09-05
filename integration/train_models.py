import pandas as pd
import numpy as np
import joblib
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from models.cnn_emotion_model import build_cnn_model
from models.lstm_vitals_model import build_lstm_model
from models.nlp_text_model import build_text_model
from integration.combined_predictor import CombinedModel

# === Load datasets ===
facial = pd.read_csv(r"C:\Users\kesav\PycharmProjects\MHD\data\FER 2013_extracted\Facial.csv")
vitals = pd.read_csv(r"C:\Users\kesav\PycharmProjects\MHD\data\vitals.csv")
text = pd.read_csv(r"C:\Users\kesav\PycharmProjects\MHD\data\text.csv")

# === Clean column names ===
facial.columns = facial.columns.str.strip().str.lower()

# === Preprocess Facial Data ===
X_facial = facial['pixels'].apply(lambda x: np.fromstring(x, sep=' ').reshape(48, 48, 1) / 255.0).values
X_facial = np.stack(X_facial)

expected_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
facial_le = LabelEncoder()
facial_le.fit(expected_classes)

facial['emotion'] = facial['emotion'].str.strip().str.lower()
facial['emotion_encoded'] = facial['emotion'].apply(
    lambda x: facial_le.transform([x])[0] if x in expected_classes else -1
)
facial = facial[facial['emotion_encoded'] != -1]
y_facial = to_categorical(facial['emotion_encoded'], num_classes=len(expected_classes))

# === Preprocess Vitals Data ===
if 'label' not in vitals.columns:
    raise ValueError("❌ Vitals CSV must contain a 'label' column.")

X_vitals = vitals.drop('label', axis=1).astype(float).values
vitals_le = LabelEncoder()
vitals_labels = vitals_le.fit_transform(vitals['label'].astype(str))
y_vitals = to_categorical(vitals_labels)
X_vitals = X_vitals.reshape((X_vitals.shape[0], X_vitals.shape[1], 1))

# === Preprocess Text Data ===
# Drop duplicate 'emotion' if exists
if 'label' in text.columns and 'emotion' not in text.columns:
    text.rename(columns={'label': 'emotion'}, inplace=True)
elif 'label' in text.columns and 'emotion' in text.columns:
    text.drop(columns=['label'], inplace=True)

# Extract emotion label from list-like or malformed data
def extract_first_label(x):
    try:
        if isinstance(x, str) and x.startswith("["):
            val = ast.literal_eval(x)
            if isinstance(val, list) and len(val) > 0:
                return str(val[0])
        elif isinstance(x, list) and len(x) > 0:
            return str(x[0])
        return str(x)
    except Exception:
        return "unknown"

text['emotion'] = text['emotion'].apply(extract_first_label)
text = text[text['emotion'] != "unknown"]
text = text[text['emotion'].notnull()]
text = text.loc[:, ~text.columns.duplicated()]

if 'text' not in text.columns:
    raise ValueError("❌ 'text' column not found in text.csv")

text_le = LabelEncoder()
text_labels = text_le.fit_transform(text['emotion'].astype(str))

X_text = text['text']
y_text = text_labels
final_labels = y_text

# === Train CNN (Facial) ===
cnn_model = build_cnn_model()
cnn_model.fit(X_facial, y_facial, epochs=25, batch_size=32, verbose=1)
cnn_preds = cnn_model.predict(X_facial)

# === Train LSTM (Vitals) ===
lstm_model = build_lstm_model(input_shape=(X_vitals.shape[1], 1))
lstm_model.fit(X_vitals, y_vitals, epochs=10, batch_size=32, verbose=1)
lstm_preds = lstm_model.predict(X_vitals)

# === Train Text Model ===
text_model = build_text_model()
text_model.fit(X_text, y_text)
text_preds = np.eye(len(np.unique(y_text)))[text_model.predict(X_text)]

# === Align predictions ===
min_len = min(len(cnn_preds), len(lstm_preds), len(text_preds), len(final_labels))
cnn_preds = cnn_preds[:min_len]
lstm_preds = lstm_preds[:min_len]
text_preds = text_preds[:min_len]
final_labels = final_labels[:min_len]

# === Train Combined Meta Model ===
meta_model = CombinedModel()
meta_model.fit(cnn_preds, lstm_preds, text_preds, final_labels)

# === Save Models and Encoders ===
cnn_model.save("models/cnn_model.keras")
lstm_model.save("models/lstm_model.h5")
joblib.dump(text_model, "models/text_model.pkl")
joblib.dump(meta_model, "models/meta_model.pkl")

joblib.dump(facial_le, "models/facial_label_encoder.pkl")
joblib.dump(vitals_le, "models/vitals_label_encoder.pkl")
joblib.dump(text_le, "models/text_label_encoder.pkl")

print("✅ All models trained and saved successfully.")


