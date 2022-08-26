import pandas as pd
import streamlit as st 
from pickle import load
import tensorflow as tf 
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow import keras

st.title('Sentiment Classifier')
st.header('Text')
Str = st.text_input('Plese write your Review')
st.sidebar.header('Description')
st.sidebar.markdown('<div style="text-align: justify;">  This web application is designed specifically for hotel review analysis. It can tell you about the sentiment behind the hotel reviews that are posted by travelers.</div>', unsafe_allow_html= True )

st.sidebar.subheader('Instructions')
st.sidebar.markdown('<div style="text-align: justify;">  Simply write or copy-paste the review in the "Text" column and hit "Get Sentiment" to see if the review is - Positive or Negative.</div>', unsafe_allow_html= True )

st.sidebar.subheader('NOTE')
st.sidebar.markdown('<div style="text-align: justify;">  Text below the emoji shows the sentiment(Positive or Negative) & the emoji illustrates the intensity of the sentiment.</div>', unsafe_allow_html= True )
review = [(Str)]
def sentiment(score):
    if pred < 0.5:
        return 'Negative'
    else:
        return 'Positive'

@st.cache(allow_output_mutation=True)
def Load_Model():
    return keras.models.load_model('./data/BERT.h5', custom_objects={'KerasLayer': hub.KerasLayer})
model = Load_Model()
pred = model.predict(review)
prediction = sentiment(pred)
#x = st.write(prediction)
button = st.button('Get Sentiment')

def image(pred):
    if pred < 0.3:
        return st.image('./data/0.png', width=100, caption='Negative')
    elif pred <= 0.6:
        return st.image('./data/1.png', width=100, caption='Negative')
    elif pred > 0.9:
        return st.image('./data/3.png', width=100, caption='Positive')
    else: 
        return st.image('./data/2.png',width=100, caption='Positive')

image(pred)