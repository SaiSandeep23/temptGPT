import spacy
nlp = spacy.load("training/models/trained/entity_recog_model")

print(nlp)
def normalize_chat_text_interaction(pinged_text):
    # Normalize the text
    pinged_text = pinged_text.lower().strip()
    print('bot_nlp => normalize_chat_text_interaction=>',pinged_text)
    return pinged_text

def process_pinged_text(text):
    print('bot_nlp => process_pinged_text 1: =>',text)
    # Process the text with the NLP object
    text = normalize_chat_text_interaction(text)
    doc = nlp(text)
    print('bot_nlp => process_pinged_text 2: =>',doc)
    # Extract the entities as strings
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print('bot_nlp => process_pinged_text 3: =>',entities)
    return entities  # Return a single string with all entities

if __name__ == "__main__":
    test_text = "Do these earbuds offer all-day battery life? "
    processed_entities = process_pinged_text(test_text)
    print(processed_entities)