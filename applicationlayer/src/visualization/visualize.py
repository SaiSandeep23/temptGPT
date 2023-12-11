from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, concatenate, TimeDistributed, RepeatVector

# Define model parameters
max_sequence_length = 100  # This is an example, the actual value will be determined by the data
vocab_size = 10000  # This is an example, the actual value will be determined by the tokenizer
embedding_dim = 256
lstm_units = 512
context_features_dim = 10  # This is an example, the actual value will be determined by the contextual data

# Define the input layers
sequence_input = Input(shape=(max_sequence_length,), dtype='int32', name='sequence_input')
context_input = Input(shape=(context_features_dim,), dtype='float32', name='context_input')

# Embedding layer for sequences
embedded_sequences = Embedding(vocab_size, embedding_dim, name='embedding_layer')(sequence_input)

# LSTM layer with return_sequences=True
lstm_layer = LSTM(lstm_units, return_sequences=True, name='lstm_layer')(embedded_sequences)

# Define a layer to process context input
context_processor = Dense(128, activation='relu', name='context_processor')(context_input)

# Repeat context to match sequence length
repeated_context = RepeatVector(max_sequence_length, name='repeat_vector')(context_processor)

# Concatenate LSTM output with repeated context
concatenated_features = concatenate([lstm_layer, repeated_context], axis=-1, name='concatenate_layer')

# Apply TimeDistributed Dense layer
dense_layer = TimeDistributed(Dense(vocab_size, activation='softmax'), name='time_distributed_layer')(concatenated_features)

# Define the model with two inputs
model = Model(inputs=[sequence_input, context_input], outputs=dense_layer)

# Now let's visualize this model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
