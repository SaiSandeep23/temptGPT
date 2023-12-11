import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bot_nlp import process_pinged_text
import pickle

class DialogueManager:
    def __init__(self, state_manager):
        model_path = './TensorFlow/tensorflowModel/chatbot_model.keras'
        tokenizer_path = './TensorFlow/tensorflowModel/tokenizer.pickle'
        self.state_manager = state_manager
        self.model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.max_sequence_length = 27

    def handle_message(self, session_id, message):
        current_state = self.state_manager.get_state(session_id)
        entities = process_pinged_text(message)
        print('entities____',entities)

        model_input = self.preprocess_for_model(entities, current_state)
        model_output = self.model.predict(model_input)
        response = self.postprocess_from_model(model_output)

        new_state = self.update_state_based_on_response(current_state, response)
        self.state_manager.update_state(session_id, new_state)
        print('e_',response,'_')
        print('final response_',response,'_')
        return response

    def preprocess_for_model(self, entities, current_state):
        entities_str = ' '.join([ent[0] for ent in entities])
        print('entities_str',entities_str)
        sequence = self.tokenizer.texts_to_sequences([entities_str])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post')

        contextual_features = self.convert_state_to_features(current_state)
        model_input = [padded_sequence, contextual_features]
        return model_input

    def postprocess_from_model(self, model_output):
        # Inspect top probabilities
        top_n = 5
        for i in range(top_n):
            word_index = np.argmax(model_output[0][i])
            probability = model_output[0][i][word_index]
            word = self.tokenizer.index_word.get(word_index, '')
            print(f"Word: {word}, Probability: {probability}")
            
        response = ''
        for word_index in np.argmax(model_output, axis=-1)[0]:
            if word_index == 0:
                break
            word = self.tokenizer.index_word.get(word_index, '')
            response += word + ' '
        print('response.strip()',response.strip())
        return response.strip()

    def update_state_based_on_response(self, current_state, response):
        new_state = dict(current_state)
        new_state['last_response'] = response
        return new_state
    
    def convert_state_to_features(self, current_state):
        features = [current_state.get(key, 3) for key in ['brand_preference', 'color_preference', 'price_range_preference', 'battery_life_preference']]
        return np.array([features])


