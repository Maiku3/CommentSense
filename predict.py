import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

VOCAB_SIZE = 10000
MAX_LEN = 250
MODEL_PATH = 'sentiment_analysis_model.h5'

# Load the saved model
model = load_model(MODEL_PATH)

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Encode the texts to sequences and pad the sequences
def encode_texts(text_list):
    sequences = tokenizer.texts_to_sequences(text_list)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, value=VOCAB_SIZE-1, padding='post')
    return padded_sequences


def predict_sentiments(text_list):
    encoded_inputs = encode_texts(text_list)
    predictions = np.argmax(model.predict(encoded_inputs), axis=-1)
    sentiments = []
    for prediction in predictions:
        if prediction == 0:
            sentiments.append("Negative")
        elif prediction == 1:
            sentiments.append("Neutral")
        else:
            sentiments.append("Positive")
    return sentiments