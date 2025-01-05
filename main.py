import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.models import load_model
import streamlit as st

word_index= imdb.get_word_index()
reverse_word_index={value: key for key, value in word_index.items()}

model =load_model("simple_rnn_imdb.h5")

## function to decode reviews
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i, "?") for i in encoded_review])
## function t process use input
def process_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) +3  for word in words]
    padded_sequence =sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_sequence

### create our prediction fuction
### Prediction function
def predict_sentiment(riview):
    processed_input =process_text(riview)

    prediction =model.predict(processed_input)

    sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment, prediction[0][0]

##Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write('Enter a movie Review To classify it as positive or negative')

user_input=st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input =process_text(user_input)

    ## Make prediction
    prediction=model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'

    ##Displat the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Predictin Score: {prediction[0][0]}')
else:
    st.write('Please Enter a movie Review')

    