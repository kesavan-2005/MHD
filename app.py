import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
import os


# === Custom CSS for Enhanced UI ===
def load_custom_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Main container styling */
    .main-header {
        background: var(--background-gradient);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0;
    }

    /* Card styling */
    .analysis-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border-left: 5px solid var(--primary-color);
    }

    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    .result-card h3 {
        margin: 0;
        font-size: 1.5rem;
    }

    .metric-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 2rem 0;
    }

    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        flex: 1;
        min-width: 200px;
        border-top: 4px solid var(--primary-color);
    }

    .metric-box h4 {
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }

    .metric-box .value {
        font-size: 2rem;
        font-weight: bold;
        color: #333;
    }

    /* Emotion labels with emojis */
    .emotion-label {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: white;
        border-radius: 25px;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    /* Button styling */
    .stButton > button {
        background: var(--background-gradient);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.3);
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }

    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid var(--primary-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: var(--background-gradient);
    }

    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed var(--primary-color);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8fafc;
    }

    /* Alert styling */
    .alert-success {
        background: linear-gradient(90deg, #10b981, #059669);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .alert-info {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


# === Enhanced Emotion Mapping ===
EMOTION_MAPPING = {
    'sadness': {'emoji': '😢', 'color': '#6366f1', 'description': 'Feeling down or melancholic'},
    'joy': {'emoji': '😊', 'color': '#10b981', 'description': 'Happy and positive emotions'},
    'love': {'emoji': '❤️', 'color': '#ef4444', 'description': 'Feelings of affection and care'},
    'anger': {'emoji': '😠', 'color': '#f59e0b', 'description': 'Frustration or irritation'},
    'fear': {'emoji': '😨', 'color': '#8b5cf6', 'description': 'Anxiety or worry'},
    'surprise': {'emoji': '😲', 'color': '#06b6d4', 'description': 'Unexpected or startling emotions'},
    'neutral': {'emoji': '😐', 'color': '#6b7280', 'description': 'Balanced emotional state'},
    'happy': {'emoji': '😄', 'color': '#10b981', 'description': 'Joyful and content'},
    'sad': {'emoji': '😢', 'color': '#6366f1', 'description': 'Feeling down or upset'},
    'angry': {'emoji': '😡', 'color': '#f59e0b', 'description': 'Irritated or frustrated'},
    'fearful': {'emoji': '😰', 'color': '#8b5cf6', 'description': 'Anxious or worried'},
    'disgusted': {'emoji': '🤢', 'color': '#84cc16', 'description': 'Feeling of revulsion'},
    'surprised': {'emoji': '😮', 'color': '#06b6d4', 'description': 'Caught off guard'}
}


# === Model Loading with Error Handling ===
@st.cache_resource
def load_models():
    """Load all models with error handling and progress tracking"""
    models = {}
    try:
        with st.spinner("Loading AI models..."):
            progress_bar = st.progress(0)

            # Base path for models (relative to app.py)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(base_dir, "models")

            # Load CNN model
            progress_bar.progress(20)
            models['cnn'] = tf.keras.models.load_model(
                os.path.join(model_dir, "cnn_model.keras")
            )

            # Load LSTM model
            progress_bar.progress(40)
            models['lstm'] = tf.keras.models.load_model(
                os.path.join(model_dir, "lstm_model.h5")
            )

            # Load text model
            progress_bar.progress(60)
            models['text'] = joblib.load(
                os.path.join(model_dir, "text_model.pkl")
            )

            # Load meta model
            progress_bar.progress(80)
            models['meta'] = joblib.load(
                os.path.join(model_dir, "meta_model.pkl")
            )

            # Load encoders
            progress_bar.progress(100)
            models['facial_le'] = joblib.load(
                os.path.join(model_dir, "facial_label_encoder.pkl")
            )
            models['vitals_le'] = joblib.load(
                os.path.join(model_dir, "vitals_label_encoder.pkl")
            )
            models['text_le'] = joblib.load(
                os.path.join(model_dir, "text_label_encoder.pkl")
            )

        st.success("✅ All models loaded successfully!")
        return models

    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None

# === Enhanced Visualization Functions ===
def create_confidence_bar_chart(probabilities, labels, title):
    """Create a horizontal bar chart showing confidence levels"""
    colors = [EMOTION_MAPPING.get(label.lower(), {'color': '#6b7280'})['color'] for label in labels]

    fig = go.Figure(data=[go.Bar(
        y=[f"{EMOTION_MAPPING.get(label.lower(), {'emoji': '🤔'})['emoji']} {label.title()}" for label in labels],
        x=probabilities,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{prob:.1%}' for prob in probabilities],
        textposition='inside'
    )])

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, color='#1f2937')),
        xaxis_title="Confidence Level",
        height=400,
        margin=dict(t=80, b=40, l=120, r=40)
    )

    return fig


def display_emotion_result(emotion, confidence=None, description=""):
    """Display emotion result with enhanced styling"""
    emotion_info = EMOTION_MAPPING.get(emotion.lower(),
                                       {'emoji': '🤔', 'color': '#6b7280', 'description': 'Unknown emotion'})

    st.markdown(f"""
    <div class="result-card" style="background: linear-gradient(135deg, {emotion_info['color']}aa, {emotion_info['color']});">
        <h3>{emotion_info['emoji']} {emotion.title()}</h3>
        <p>{description or emotion_info['description']}</p>
        {f'<p><strong>Confidence: {confidence:.1%}</strong></p>' if confidence else ''}
    </div>
    """, unsafe_allow_html=True)


# === Page Configuration ===
st.set_page_config(
    page_title="MindScan: AI Mental Health Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_custom_css()

# === Main Header ===
st.markdown("""
<div class="main-header">
    <h1>🧠 MindScan</h1>
    <p>AI-Powered Mental Health Detection System</p>
    <p>Analyze emotions through facial expressions, vital signs, and text sentiment</p>
</div>
""", unsafe_allow_html=True)

# === Load Models ===
models = load_models()
if not models:
    st.stop()

# === Sidebar Navigation ===
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
    <h3 style="color: white; margin: 0;">🔍 Analysis Modes</h3>
</div>
""", unsafe_allow_html=True)

option = st.sidebar.radio(
    "Choose your analysis method:",
    ["🏠 Dashboard", "📷 Facial Analysis", "💓 Vital Signs", "📝 Text Analysis", "🎥 Live Camera", "📼 Video Upload",
     "🔮 Combined Analysis"],
    index=0
)

# === Dashboard Overview ===
if option == "🏠 Dashboard":
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.header("🎯 Welcome to MindScan")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-box">
            <h4>📷 Facial Analysis</h4>
            <div class="value">CNN</div>
            <p>Advanced facial emotion recognition using deep learning</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-box">
            <h4>💓 Vital Signs</h4>
            <div class="value">LSTM</div>
            <p>Emotion detection from physiological data</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-box">
            <h4>📝 Text Analysis</h4>
            <div class="value">NLP</div>
            <p>Sentiment analysis from written text</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Emotion Reference Guide
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.header("🎭 Emotion Reference Guide")

    cols = st.columns(4)
    emotions = list(EMOTION_MAPPING.keys())

    for i, emotion in enumerate(emotions[:8]):  # Show first 8 emotions
        with cols[i % 4]:
            emotion_info = EMOTION_MAPPING[emotion]
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; margin: 0.5rem 0; background: {emotion_info['color']}20; border-radius: 10px; border: 2px solid {emotion_info['color']}40;">
                <div style="font-size: 2rem;">{emotion_info['emoji']}</div>
                <div style="font-weight: bold; color: {emotion_info['color']};">{emotion.title()}</div>
                <div style="font-size: 0.8rem; color: #666;">{emotion_info['description']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# === Facial Image Analysis ===
elif option == "📷 Facial Analysis":
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.header("📷 Facial Emotion Detection")
    st.markdown("Upload a clear facial image for emotion analysis using our advanced CNN model.")

    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear facial image (preferably 48x48 pixels or larger)"
    )

    if uploaded_image:
        col1, col2 = st.columns([1, 2])

        with col1:
            image = Image.open(uploaded_image).convert('L').resize((48, 48))
            st.image(image, caption="Uploaded Image", width=200)

            # Process image
            with st.spinner("Analyzing facial emotion..."):
                img_array = np.array(image).reshape(1, 48, 48, 1) / 255.0
                cnn_pred = models['cnn'].predict(img_array)
                facial_index = np.argmax(cnn_pred)
                facial_status = models['facial_le'].inverse_transform([facial_index])[0]
                confidence = np.max(cnn_pred)

        with col2:
            display_emotion_result(facial_status, confidence, "Detected from facial expression analysis")

            # Show probability distribution
            labels = models['facial_le'].classes_
            fig = create_confidence_bar_chart(cnn_pred[0], labels, "Emotion Confidence Levels")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👆 Please upload a facial image to begin analysis")

    st.markdown("</div>", unsafe_allow_html=True)

# === Vital Signs Analysis ===
elif option == "💓 Vital Signs":
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.header("💓 Emotion Analysis from Vital Signs")
    st.markdown("Adjust the sliders to match your current vital signs for emotion prediction.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📊 Input Your Vital Signs")
        heart_rate = st.slider("❤️ Heart Rate (bpm)", 50, 150, 80, help="Normal resting: 60-100 bpm")
        respiratory_rate = st.slider("🫁 Respiratory Rate (breaths/min)", 10, 40, 20, help="Normal: 12-20 breaths/min")
        spo2 = st.slider("🩸 SpO2 (%)", 70, 100, 98, help="Normal: 95-100%")
        temp = st.slider("🌡️ Body Temperature (°C)", 35.0, 40.0, 36.5, help="Normal: 36.1-37.2°C")

        # Vital signs status indicators
        st.markdown("### 📈 Vital Signs Status")

        # Heart rate status
        if 60 <= heart_rate <= 100:
            hr_status = "🟢 Normal"
        elif heart_rate < 60:
            hr_status = "🔵 Bradycardia"
        else:
            hr_status = "🔴 Tachycardia"

        # SpO2 status
        if spo2 >= 95:
            spo2_status = "🟢 Normal"
        elif spo2 >= 90:
            spo2_status = "🟡 Mild hypoxemia"
        else:
            spo2_status = "🔴 Severe hypoxemia"

        st.markdown(f"- **Heart Rate**: {hr_status}")
        st.markdown(f"- **SpO2**: {spo2_status}")

    with col2:
        # Process vital signs
        vitals_input = np.array([[heart_rate, respiratory_rate, spo2, temp]]).reshape((1, 4, 1))

        with st.spinner("Analyzing vital signs..."):
            lstm_pred = models['lstm'].predict(vitals_input)
            vitals_index = np.argmax(lstm_pred)
            vitals_status = models['vitals_le'].inverse_transform([vitals_index])[0]
            confidence = np.max(lstm_pred)

        st.markdown("### 📊 Analysis Result")

        # Simple text display instead of the card
        emotion_info = EMOTION_MAPPING.get(vitals_status.lower(), {'emoji': '🤔', 'color': '#6b7280'})
        st.markdown(f"**Detected Emotion:** {emotion_info['emoji']} {vitals_status.title()}")
        st.markdown(f"**Confidence:** {confidence:.1%}")

        # Show bar chart
        labels = models['vitals_le'].classes_
        fig = create_confidence_bar_chart(lstm_pred[0], labels, "Emotion Confidence Levels")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# === Text Analysis ===
elif option == "📝 Text Analysis":
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.header("📝 Text Emotion Analysis")
    st.markdown("Share your thoughts and let our NLP model analyze the emotional content.")

    # Sample prompts
    sample_prompts = [
        "I'm feeling really excited about my new job opportunity!",
        "Today has been quite challenging and I'm feeling overwhelmed.",
        "I love spending time with my family on weekends.",
        "I'm worried about the upcoming presentation at work.",
        "This beautiful sunset makes me feel so peaceful."
    ]

    st.markdown("### 💭 Quick Start - Try These Examples:")
    cols = st.columns(len(sample_prompts))
    for i, prompt in enumerate(sample_prompts):
        if cols[i].button(f"Example {i + 1}", key=f"sample_{i}"):
            st.session_state.text_input = prompt

    text_input = st.text_area(
        "✍️ Describe your current mental state or feelings:",
        value=st.session_state.get('text_input', ''),
        height=150,
        placeholder="Express your thoughts, feelings, or current situation here..."
    )

    if text_input:
        col1, col2 = st.columns([1, 1])

        with col1:
            with st.spinner("Analyzing text sentiment..."):
                text_pred_class = models['text'].predict([text_input])[0]
                text_status = models['text_le'].inverse_transform([text_pred_class])[0]

                # Get prediction probabilities (if available)
                try:
                    text_pred_proba = models['text'].predict_proba([text_input])[0]
                    confidence = np.max(text_pred_proba)
                except:
                    confidence = None

            display_emotion_result(text_status, confidence, "Detected from text sentiment analysis")

            # Text statistics
            st.markdown("### 📊 Text Statistics")
            word_count = len(text_input.split())
            char_count = len(text_input)

            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-box">
                    <h4>Word Count</h4>
                    <div class="value">{word_count}</div>
                </div>
                <div class="metric-box">
                    <h4>Character Count</h4>
                    <div class="value">{char_count}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Show emotion breakdown if probabilities available
            if 'text_pred_proba' in locals():
                labels = models['text_le'].classes_
                fig = create_confidence_bar_chart(text_pred_proba, labels, "Emotion Confidence Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    "💡 Text analysis complete! The model has classified your text based on emotional content patterns.")

    else:
        st.info("👆 Please enter some text to analyze its emotional content")

    st.markdown("</div>", unsafe_allow_html=True)

# === Live Camera Analysis ===
elif option == "🎥 Live Camera":
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.header("🎥 Real-Time Facial Emotion Detection")
    st.markdown(
        "Use your camera for live emotion detection. Make sure you have good lighting and your face is clearly visible.")


    class EmotionDetector(VideoTransformerBase):
        def __init__(self):
            self.emotion_history = []

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            try:
                resized = cv2.resize(gray, (48, 48)) / 255.0
                input_img = resized.reshape(1, 48, 48, 1)
                pred = models['cnn'].predict(input_img)
                label = models['facial_le'].inverse_transform([np.argmax(pred)])[0]
                confidence = np.max(pred)

                # Get emoji for emotion
                emoji = EMOTION_MAPPING.get(label.lower(), {'emoji': '🤔'})['emoji']

                # Draw results on frame
                cv2.putText(img, f"{emoji} {label.title()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(img, f"Confidence: {confidence:.1%}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                            2)

                # Store emotion history
                self.emotion_history.append(label)
                if len(self.emotion_history) > 30:  # Keep last 30 detections
                    self.emotion_history.pop(0)

            except Exception as e:
                cv2.putText(img, "Processing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            return img


    # Instructions
    st.markdown("""
    ### 📋 Instructions:
    1. Click "START" to begin live emotion detection
    2. Ensure good lighting and position your face in the frame
    3. The system will analyze your emotions in real-time
    4. Click "STOP" when finished
    """)

    webrtc_streamer(
        key="live-emotion-detection",
        video_transformer_factory=EmotionDetector,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    st.markdown("</div>", unsafe_allow_html=True)

# === Video Upload Analysis ===
elif option == "📼 Video Upload":
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.header("📼 Video Emotion Analysis")
    st.markdown("Upload a video file for comprehensive emotion analysis across all frames.")

    video_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov'],
        help="Upload a video file for frame-by-frame emotion analysis"
    )

    if video_file:
        # Save uploaded video
        with open("temp_video.mp4", 'wb') as f:
            f.write(video_file.read())

        col1, col2 = st.columns([1, 1])

        with col1:
            st.video("temp_video.mp4")

        with col2:
            if st.button("🎬 Analyze Video", type="primary"):
                with st.spinner("Processing video frames..."):
                    cap = cv2.VideoCapture("temp_video.mp4")
                    frame_labels = []
                    frame_confidences = []
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    progress_bar = st.progress(0)
                    frame_count = 0

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        try:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            resized = cv2.resize(gray, (48, 48)) / 255.0
                            input_img = resized.reshape(1, 48, 48, 1)
                            pred = models['cnn'].predict(input_img)
                            label = models['facial_le'].inverse_transform([np.argmax(pred)])[0]
                            confidence = np.max(pred)

                            frame_labels.append(label)
                            frame_confidences.append(confidence)

                        except:
                            continue

                        frame_count += 1
                        progress_bar.progress(frame_count / total_frames)

                    cap.release()

                if frame_labels:
                    # Calculate statistics
                    most_common = max(set(frame_labels), key=frame_labels.count)
                    avg_confidence = np.mean(frame_confidences)

                    display_emotion_result(most_common, avg_confidence,
                                           f"Dominant emotion across {len(frame_labels)} analyzed frames")

                    # Video analysis summary
                    st.markdown("### 📊 Video Analysis Summary")
                    unique_emotions = len(set(frame_labels))
                    dominant_percentage = (frame_labels.count(most_common) / len(frame_labels)) * 100

                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-box">
                            <h4>Total Frames</h4>
                            <div class="value">{len(frame_labels)}</div>
                        </div>
                        <div class="metric-box">
                            <h4>Unique Emotions</h4>
                            <div class="value">{unique_emotions}</div>
                        </div>
                        <div class="metric-box">
                            <h4>Dominance</h4>
                            <div class="value">{dominant_percentage:.1f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("❌ No faces detected in the video. Please ensure the video contains clear facial images.")

    else:
        st.info("👆 Please upload a video file to begin analysis")

    st.markdown("</div>", unsafe_allow_html=True)

# === Combined Analysis ===
elif option == "🔮 Combined Analysis":
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.header("🔮 Comprehensive Mental Health Analysis")
    st.markdown("Combine multiple analysis methods for the most accurate mental health assessment.")

    # Initialize session state for storing predictions
    if 'predictions' not in st.session_state:
        st.session_state.predictions = {}

    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📷 Facial Input", "💓 Vital Signs", "📝 Text Input"])

    with tab1:
        st.markdown("### Upload a facial image")
        facial_image = st.file_uploader("Choose facial image", type=['png', 'jpg', 'jpeg'], key="combined_facial")

        if facial_image:
            col1, col2 = st.columns([1, 2])
            with col1:
                image = Image.open(facial_image).convert('L').resize((48, 48))
                st.image(image, caption="Facial Input", width=150)

            with col2:
                if st.button("🔍 Analyze Face", key="analyze_face"):
                    with st.spinner("Processing facial emotion..."):
                        img_array = np.array(image).reshape(1, 48, 48, 1) / 255.0
                        cnn_pred = models['cnn'].predict(img_array)
                        facial_index = np.argmax(cnn_pred)
                        facial_status = models['facial_le'].inverse_transform([facial_index])[0]

                        st.session_state.predictions['facial'] = {
                            'prediction': cnn_pred,
                            'label': facial_status,
                            'confidence': np.max(cnn_pred)
                        }

                        display_emotion_result(facial_status, np.max(cnn_pred), "From facial analysis")

    with tab2:
        st.markdown("### Input your vital signs")
        col1, col2 = st.columns(2)

        with col1:
            heart_rate = st.slider("❤️ Heart Rate", 50, 150, 80, key="combined_hr")
            respiratory_rate = st.slider("🫁 Respiratory Rate", 10, 40, 20, key="combined_rr")

        with col2:
            spo2 = st.slider("🩸 SpO2", 70, 100, 98, key="combined_spo2")
            temp = st.slider("🌡️ Temperature", 35.0, 40.0, 36.5, key="combined_temp")

        if st.button("💓 Analyze Vitals", key="analyze_vitals"):
            with st.spinner("Processing vital signs..."):
                vitals_input = np.array([[heart_rate, respiratory_rate, spo2, temp]]).reshape((1, 4, 1))
                lstm_pred = models['lstm'].predict(vitals_input)
                vitals_index = np.argmax(lstm_pred)
                vitals_status = models['vitals_le'].inverse_transform([vitals_index])[0]

                st.session_state.predictions['vitals'] = {
                    'prediction': lstm_pred,
                    'label': vitals_status,
                    'confidence': np.max(lstm_pred)
                }

                display_emotion_result(vitals_status, np.max(lstm_pred), "From vital signs analysis")

    with tab3:
        st.markdown("### Enter your thoughts")
        text_input = st.text_area("Describe your feelings", height=120, key="combined_text")

        if text_input and st.button("📝 Analyze Text", key="analyze_text"):
            with st.spinner("Processing text sentiment..."):
                text_pred_class = models['text'].predict([text_input])[0]
                text_status = models['text_le'].inverse_transform([text_pred_class])[0]

                # Create one-hot encoded prediction for meta model
                text_pred = np.zeros((1, len(models['text_le'].classes_)))
                text_pred[0, text_pred_class] = 1

                st.session_state.predictions['text'] = {
                    'prediction': text_pred,
                    'label': text_status,
                    'confidence': 1.0  # Assuming high confidence for classification
                }

                display_emotion_result(text_status, 1.0, "From text sentiment analysis")

    # Combined prediction section
    st.markdown("---")
    st.markdown("### 🎯 Final Combined Analysis")

    if len(st.session_state.predictions) > 0:
        # Show individual predictions summary
        st.markdown("#### 📋 Individual Analysis Results:")
        cols = st.columns(len(st.session_state.predictions))

        for i, (method, pred_data) in enumerate(st.session_state.predictions.items()):
            with cols[i]:
                method_names = {'facial': '📷 Facial', 'vitals': '💓 Vitals', 'text': '📝 Text'}
                st.markdown(f"""
                <div class="metric-box">
                    <h4>{method_names[method]}</h4>
                    <div class="value">{EMOTION_MAPPING.get(pred_data['label'].lower(), {'emoji': '🤔'})['emoji']}</div>
                    <p>{pred_data['label'].title()}</p>
                    <small>Confidence: {pred_data['confidence']:.1%}</small>
                </div>
                """, unsafe_allow_html=True)

        # Generate combined prediction if we have all three modalities
        if len(st.session_state.predictions) >= 2:
            if st.button("🧠 Generate Final Prediction", type="primary", key="final_prediction"):
                with st.spinner("Computing combined mental health assessment..."):
                    try:
                        # Prepare features for meta model
                        feature_list = []

                        if 'facial' in st.session_state.predictions:
                            feature_list.append(st.session_state.predictions['facial']['prediction'])
                        else:
                            # Use neutral prediction if facial not available
                            neutral_pred = np.zeros((1, len(models['facial_le'].classes_)))
                            neutral_pred[0, len(models['facial_le'].classes_) // 2] = 1
                            feature_list.append(neutral_pred)

                        if 'vitals' in st.session_state.predictions:
                            feature_list.append(st.session_state.predictions['vitals']['prediction'])
                        else:
                            # Use neutral prediction if vitals not available
                            neutral_pred = np.zeros((1, len(models['vitals_le'].classes_)))
                            neutral_pred[0, len(models['vitals_le'].classes_) // 2] = 1
                            feature_list.append(neutral_pred)

                        if 'text' in st.session_state.predictions:
                            feature_list.append(st.session_state.predictions['text']['prediction'])
                        else:
                            # Use neutral prediction if text not available
                            neutral_pred = np.zeros((1, len(models['text_le'].classes_)))
                            neutral_pred[0, len(models['text_le'].classes_) // 2] = 1
                            feature_list.append(neutral_pred)

                        # Concatenate features
                        features = np.concatenate(feature_list, axis=1)

                        # Make final prediction
                        final_pred = models['meta'].predict(features)[0]
                        final_label = models['text_le'].inverse_transform([final_pred])[0]

                        # Calculate confidence based on agreement between modalities
                        labels = [pred_data['label'] for pred_data in st.session_state.predictions.values()]
                        agreement = len(set(labels)) / len(labels)  # Higher agreement = lower diversity
                        confidence = 1.0 - agreement + 0.5  # Adjust confidence based on agreement
                        confidence = min(confidence, 1.0)

                        # Display final result
                        st.markdown("#### 🎯 Final Mental Health Assessment")
                        display_emotion_result(final_label, confidence, "Comprehensive analysis result")

                        # Detailed breakdown
                        st.markdown("#### 📊 Analysis Breakdown")

                        # Create comparison chart
                        methods = list(st.session_state.predictions.keys())
                        emotions = [st.session_state.predictions[method]['label'] for method in methods]
                        confidences = [st.session_state.predictions[method]['confidence'] for method in methods]

                        fig = go.Figure(data=[
                            go.Bar(
                                x=[m.title() for m in methods],
                                y=confidences,
                                text=[f"{EMOTION_MAPPING.get(e.lower(), {'emoji': '🤔'})['emoji']} {e.title()}" for e in
                                      emotions],
                                textposition='inside',
                                marker=dict(
                                    color=[EMOTION_MAPPING.get(e.lower(), {'color': '#6b7280'})['color'] for e in
                                           emotions])
                            )
                        ])

                        fig.update_layout(
                            title="Analysis Method Comparison",
                            yaxis_title="Confidence Level",
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Recommendations based on final result
                        st.markdown("#### 💡 Recommendations")
                        recommendations = {
                            'sadness': "Consider engaging in activities that bring you joy, connecting with supportive friends, or speaking with a mental health professional.",
                            'joy': "Great! Maintain this positive state through continued self-care and healthy habits.",
                            'love': "Wonderful! Cherish these positive feelings and consider sharing them with others.",
                            'anger': "Try relaxation techniques like deep breathing, physical exercise, or talking through your feelings.",
                            'fear': "Consider grounding techniques, gradual exposure to fears, or seeking support from trusted individuals.",
                            'surprise': "Take time to process unexpected events and adapt to new situations at your own pace.",
                            'neutral': "A balanced state. Consider activities that promote emotional well-being and personal growth."
                        }

                        recommendation = recommendations.get(final_label.lower(),
                                                             "Consider maintaining awareness of your emotional state and practicing self-care.")

                        st.info(f"💭 **Personalized Suggestion**: {recommendation}")

                        # Disclaimer
                        st.warning(
                            "⚠️ **Important**: This analysis is for informational purposes only and should not replace professional medical or psychological advice. If you're experiencing persistent mental health concerns, please consult with a qualified healthcare provider.")

                    except Exception as e:
                        st.error(f"❌ Error generating combined prediction: {str(e)}")
                        st.info("Please ensure you have completed at least two types of analysis above.")
        else:
            st.info("📝 Complete at least two analysis types above to generate a comprehensive assessment.")

    else:
        st.info("👆 Start by completing one or more analysis types in the tabs above.")

    # Reset predictions button
    if st.session_state.predictions:
        if st.button("🔄 Start New Analysis", key="reset_predictions"):
            st.session_state.predictions = {}
            st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8fafc; border-radius: 10px; margin-top: 2rem;">
    <h4>🧠 MindScan - AI Mental Health Detection</h4>
    <p>Powered by advanced machine learning algorithms including CNN, LSTM, and NLP models</p>
    <p><small>⚠️ This tool is for informational purposes only. Please consult healthcare professionals for medical advice.</small></p>
</div>
""", unsafe_allow_html=True)


