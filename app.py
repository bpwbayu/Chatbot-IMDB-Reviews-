import streamlit as st
import pickle
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z]', ' ', text)
    words = [w for w in text.split() if w not in stop]
    return ' '.join(words)



try:
    model = joblib.load("model_logreg.joblib")
    tfidf = joblib.load("tfidf_vectorizer.joblib")
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(f"Gagal memuat model atau TF-IDF: {e}")

def predict_sentiment(text):
    clean = preprocess_text(text)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    sentiment = "Positive 😊" if pred == 1 else "Negative 😞"
    return sentiment

st.set_page_config(page_title="Sentiment Analysis IMDB", page_icon="🎬")
st.title("🎬 Sentiment Analysis - IMDB Movie Reviews")
st.write("Type your film review below to find out whether the sentiment is **positive** or **negative**!")

st.divider()

user_input = st.text_area("📝 Type your film review here:")

if st.button("🔍 Predict"):
    if not model_ready:
        st.error("The model is not ready. Ensure that the model_logreg.pkl and tfidf_vectorizer.pkl files are in the same folder.")
    elif user_input.strip():
        hasil = predict_sentiment(user_input)

        if "Positive" in hasil:
            st.markdown(
                f"""
                <div style="background-color:#0f5132;padding:15px;border-radius:10px;color:white;text-align:center;">
                    <strong></strong> {hasil}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="background-color:#842029;padding:15px;border-radius:10px;color:white;text-align:center;">
                    <strong></strong> {hasil}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("Please enter the text first.")