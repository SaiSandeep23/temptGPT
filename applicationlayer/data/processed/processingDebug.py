import spacy
from spacy.training import offsets_to_biluo_tags

nlp = spacy.blank('en')  # or load a specific model

def check_alignment(text, entities):
    doc = nlp.make_doc(text)
    biluo_tags = offsets_to_biluo_tags(doc, entities)
    misaligned_entities = []
    for ent, tag in zip(entities, biluo_tags):
        if '-' in tag:
            misaligned_entities.append((ent, tag))
        print(f"Entity: {ent}, Token: {doc.char_span(ent[0], ent[1])}, Tag: {tag}")
    return misaligned_entities

def print_token_boundaries(text):
    doc = nlp.make_doc(text)
    for token in doc:
        print(f"Token: '{token.text}', Start: {token.idx}, End: {token.idx + len(token)}")

# Example usage to print token boundaries
text = "Does it come with additional earbuds and a case?"
print_token_boundaries(text)

# Assuming we have an entity "additional earbuds and a case" starting at index 22 and ending at index 43
entities = [(18, 47, 'ACCESSORIES')]  # Replace with your actual entity spans
misaligned_entities = check_alignment(text, entities)
print("Misaligned entities:", misaligned_entities)
