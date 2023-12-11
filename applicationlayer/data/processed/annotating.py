import spacy

nlp = spacy.blank("en")

ACCESSORIES = [
    ("Does it come with additional earbuds and a case?", {"entities": [(22, 33, "ACCESSORIES")]}),
    ("Are extra foam tips included with these earphones?", {"entities": [(7, 18, "ACCESSORIES")]}),
    # Add more sentences here
]

corrected_tuples = []

for text, annotations in ACCESSORIES:
    corrected_entities = []
    doc = nlp(text)
    token_start = 0

    for start, end, label in annotations['entities']:
        entity_text = text[start:end]
        
        # Find the start and end indices of the entity in tokenized text
        entity_start = None
        entity_end = None
        
        for token in doc:
            if token_start <= start < token_start + len(token.text):
                entity_start = token.idx + start - token_start
            if token_start <= end <= token_start + len(token.text):
                entity_end = token.idx + end - token_start
                break

            token_start += len(token.text)

        if entity_start is not None and entity_end is not None:
            corrected_entities.append((entity_start, entity_end, label))

    corrected_annotation = {"entities": corrected_entities}
    corrected_tuples.append((text, corrected_annotation))

# Print the corrected tuples
for tuple_data in corrected_tuples:
    print(tuple_data)
