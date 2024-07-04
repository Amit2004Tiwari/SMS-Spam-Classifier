import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
ps = PorterStemmer()
import scipy
import numpy as np


nltk.download('stopwords')
nltk.download('punkt')

def transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


custom_css = """
            <style>
            
            footer {visibility: hidden;}
            header {visibility: hidden;}
            body {
                background-color: 
            }
            .main {
                padding: 20px;
                border-radius: 10px;
            }
            h1 {
                color: 
            }
            .footer {
                font-size: 0.9em;
                margin-top: 50px;
                text-align: center;
            }
            </style>
            """

st.markdown(custom_css, unsafe_allow_html=True)


st.title("📩 SMS Spam Classifier")



input_sms = st.text_area("Enter the message", height=200)


if st.button('Predict'):
    with st.spinner('Analyzing...'):
        
        transform_sms = transform(input_sms)

        
        vector_input = tfidf.transform([transform_sms])

        
        result = model.predict(vector_input)[0]

        
        if result == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")


st.markdown(
    """
    <div class="footer">
        Created by Amit Tiwari
    </div>
    """, 
    unsafe_allow_html=True
)
