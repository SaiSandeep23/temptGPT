import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, concatenate, TimeDistributed, RepeatVector
import pickle

# Load dialogue pairs and context data
with open('conversationalPairs/dialoguePairOne.json', 'r') as f:
    dialogue_pairs = json.load(f)

with open('dataPreparation/contextual_data.json', 'r') as f:
    contextual_data = json.load(f)

# Create tokenizer and fit on the dialogue pairs
tokenizer = Tokenizer()
tokenizer.fit_on_texts([pair['user_message'] for pair in dialogue_pairs] +
                       [pair['bot_response'] for pair in dialogue_pairs])

# Convert text to sequences of integers
user_sequences = tokenizer.texts_to_sequences([pair['user_message'] for pair in dialogue_pairs])
response_sequences = tokenizer.texts_to_sequences([pair['bot_response'] for pair in dialogue_pairs])

# Find the maximum sequence length for padding
max_sequence_length = max(len(seq) for seq in user_sequences + response_sequences)
padded_user_sequences = pad_sequences(user_sequences, maxlen=max_sequence_length, padding='post')
padded_response_sequences = pad_sequences(response_sequences, maxlen=max_sequence_length, padding='post')

# Prepare context data
contextual_features = np.array([
    [float(x) if x is not None else 3 for x in row] for row in zip(*contextual_data.values())
])

# Define model parameters
vocab_size = len(tokenizer.word_index) + 1  # Plus one for padding token
embedding_dim = 256
lstm_units = 512

print('tokenizer.word_index=>',tokenizer.word_index)

# Define the input layers
sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
context_input = Input(shape=(4,), dtype='float32')  
# Embedding layer for sequences
embedded_sequences = Embedding(vocab_size, embedding_dim)(sequence_input)

# LSTM layer with return_sequences=True
lstm_layer = LSTM(lstm_units, return_sequences=True)(embedded_sequences)

# Define a layer to process context input
context_processor = Dense(128, activation='relu')(context_input)

# Repeat context to match sequence length
repeated_context = RepeatVector(max_sequence_length)(context_processor)

# Concatenate LSTM output with repeated context
concatenated_features = concatenate([lstm_layer, repeated_context], axis=-1)

# Apply TimeDistributed Dense layer
dense_layer = TimeDistributed(Dense(vocab_size, activation='softmax'))(concatenated_features)

# Define the model with two inputs
model = Model(inputs=[sequence_input, context_input], outputs=dense_layer)

# One-hot encode response sequences
def one_hot_encode(sequences, vocab_size):
    results = np.zeros((len(sequences), max_sequence_length, vocab_size))
    for i, sequence in enumerate(sequences):
        for t, word_index in enumerate(sequence):
            results[i, t, word_index] = 1.
    return results

one_hot_response_sequences = one_hot_encode(padded_response_sequences, vocab_size)

# Compile the model - Change loss function to 'categorical_crossentropy'
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Print shapes for debugging
print("User sequences shape:", padded_user_sequences.shape)
print("Response sequences shape:", padded_response_sequences.shape)
print("Contextual features shape:", contextual_features.shape)
print("vocab_size:", vocab_size)

# Print model summary for additional debugging
model.summary()

# Perform a test prediction for debugging
sample_output = model.predict([padded_user_sequences[:2], contextual_features[:2]])
print("Sample output shape:", sample_output.shape)

# Train the model
try:
    model.fit(
        [padded_user_sequences, contextual_features],
        one_hot_response_sequences,
        epochs=10
    )
except Exception as e:
    print("Error during model training:", e)

# Save the trained model
model.save('tensorflowModel/chatbot_model.keras')

# Save the tokenizer
with open('tensorflowModel/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
