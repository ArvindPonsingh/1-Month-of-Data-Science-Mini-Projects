import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


texts = [
    "This is a great product",
    "I did not like the movie",
    "Amazing experience",
    "Not good, would not recommend",
    "It was okay, not the best",
]

labels = [1, 0, 1, 0, 0]  


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)


X = pad_sequences(sequences, maxlen=10)
y = np.array(labels)

# Define the CNN model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=10))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()

model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
