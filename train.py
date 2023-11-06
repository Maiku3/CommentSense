import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, GlobalAveragePooling1D,  Dropout, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd

# Parameters
VOCAB_SIZE = 10000
MAX_LEN = 250
EMBEDDING_DIM = 16
MODEL_PATH = 'sentiment_analysis_model.keras'

file_path = 'training_data.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')
df_shuffled = data.sample(frac=1).reset_index(drop=True)

texts = []
labels = []

# Process the data from the csv file with text and label columns
for _, row in df_shuffled.iterrows():
    texts.append(row[-1])
    label = row[0]
    labels.append(0 if label == 0 else 1 if label == 2 else 2) # 0 = negative, 1 = neutral, 2 = positive

texts = np.array(texts)
labels = np.array(labels)

# Tokenize the data
tokenizer = keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, value=VOCAB_SIZE-1, padding='post')

# Save the tokenizer to a file
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Split the data into training and testing
split = int(0.8 * len(texts))
x_train = padded_sequences[:split]  # 80% training data
x_val = padded_sequences[split:]  # 20% testing data
y_train = labels[:split]
y_val = labels[split:]

# Check if saved model exists
if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Training a new model...")
    # Define the model to classify text into 3 categories (negative, neutral, positive)
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        Dropout(0.2),
        GlobalAveragePooling1D(),
        Dropout(0.2),
        Dense(24, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

    # Save the trained model
    model.save(MODEL_PATH)

# Evaluate the model
loss, accuracy = model.evaluate(x_val, y_val)
print("Loss:", loss)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Interactive loop for predictions
def encode_text(text):
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [tokenizer.word_index[word] if word in tokenizer.word_index else 0 for word in tokens]
    return pad_sequences([tokens], maxlen=MAX_LEN, padding='post', value=VOCAB_SIZE-1)

while True:
    user_input = input("Enter a sentence for sentiment analysis (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break
    
    encoded_input = encode_text(user_input)
    prediction = np.argmax(model.predict(encoded_input))

    if prediction == 0:
        print("Sentiment: Negative")
    elif prediction == 1:
        print("Sentiment: Neutral")
    else:
        print("Sentiment: Positive")