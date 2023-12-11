import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Define patterns
patterns = {
    "ACCESSORIES": [
        [{"LOWER": "additional"}, {"LOWER": "earbuds"}, {"LOWER": "and"}, {"LOWER": "a"}, {"LOWER": "case"}],
        [{"LOWER": "extra"}, {"LOWER": "foam"}, {"LOWER": "tips"}],
        [{"LOWER": "carrying"}, {"LOWER": "pouch"}],
        [{"LOWER": "replaceable"}, {"LOWER": "cable"}],
        [{"LOWER": "wireless"}, {"LOWER": "charging"}, {"LOWER": "case"}],
        [{"LOWER": "customizable"}, {"LOWER": "ear"}, {"LOWER": "tips"}],
        [{"LOWER": "detachable"}, {"LOWER": "microphone"}],
        [{"LOWER": "different"}, {"LOWER": "sized"}, {"LOWER": "silicone"}, {"LOWER": "tips"}],
        [{"LOWER": "audio"}, {"LOWER": "cable"}, {"LOWER": "extension"}],
        [{"LOWER": "protective"}, {"LOWER": "case"}],
        [{"LOWER": "spare"}, {"LOWER": "parts"}, {"LOWER": "like"}, {"LOWER": "ear"}, {"LOWER": "pads"}],
        [{"LOWER": "travel"}, {"LOWER": "adapter"}],
        [{"LOWER": "usb"}, {"LOWER": "charging"}, {"LOWER": "cable"}],
        [{"LOWER": "headphone"}, {"LOWER": "stand"}],
        [{"LOWER": "bluetooth"}, {"LOWER": "transmitters"}],
        [{"LOWER": "warranty"}, {"LOWER": "card"}, {"LOWER": "and"}, {"LOWER": "a"}, {"LOWER": "user"}, {"LOWER": "manual"}],
        [{"LOWER": "airplane"}, {"LOWER": "adapter"}],
        [{"LOWER": "bass"}, {"LOWER": "boosters"}],
        [{"LOWER": "portable"}, {"LOWER": "charger"}],
        [{"LOWER": "extended"}, {"LOWER": "warranty"}, {"LOWER": "and"}, {"LOWER": "extra"}, {"LOWER": "cables"}],
        [{"LOWER": "noise-canceling"}, {"LOWER": "modules"}],
        [{"LOWER": "3.5mm"}, {"LOWER": "jack"}, {"LOWER": "adapter"}],
        [{"LOWER": "custom"}, {"LOWER": "ear"}, {"LOWER": "molds"}],
        [{"LOWER": "detachable"}, {"LOWER": "voice"}, {"LOWER": "mic"}],
        [{"LOWER": "extra-long"}, {"LOWER": "cords"}],
        [{"LOWER": "changeable"}, {"LOWER": "decorative"}, {"LOWER": "plates"}],
        [{"LOWER": "neoprene"}, {"LOWER": "sleeve"}],
        [{"LOWER": "volume-limiting"}, {"LOWER": "adapters"}],
        [{"LOWER": "sweat-resistant"}, {"LOWER": "covers"}],
        [{"LOWER": "active"}, {"LOWER": "noise-canceling"}, {"LOWER": "modules"}],
        [{"LOWER": "gold-plated"}, {"LOWER": "audio"}, {"LOWER": "connector"}]
    ]
}

# Add patterns to the matcher
for label, label_patterns in patterns.items():
    matcher.add(label, label_patterns)

def prepare_training_data(sentences):
    training_data = []
    for sentence in sentences:
        doc = nlp(sentence)
        matches = matcher(doc)
        entities = []
        for match_id, start, end in matches:
            span = doc[start:end]  # The matched span
            entities.append((span.start_char, span.end_char, nlp.vocab.strings[match_id]))
        training_data.append((sentence, {'entities': entities}))
    return training_data

# Example sentences
sentences = [
    "Does it come with additional earbuds and a case?",
    "Are extra foam tips included with these earphones?",
    "Do these headphones come with a carrying pouch?",
    "I'm looking for headphones with a replaceable cable.",
    'Does this model include a wireless charging case?',
    'Are there any headphones with customizable ear tips?',
    'I need headphones that have a detachable microphone.',
    'Do these earbuds come with different sized silicone tips?',
    'Looking for headphones that offer an audio cable extension.',
    'Can I get a protective case for these headphones?',
    'Do you provide spare parts like ear pads for these headphones?',
    'Is a travel adapter included with these over-ear headphones?',
    'Are these headphones sold with a USB charging cable?',
    'Does the package include a headphone stand?',
    "I'm interested in headphones with Bluetooth transmitters.",
    'Do they come with a warranty card and a user manual?',
    'Is there an airplane adapter available with this model?',
    'Looking for headphones with additional bass boosters.',
    'Do these wireless headphones include a portable charger?',
    'Can you add an extended warranty and extra cables to my order?',
    'Are noise-canceling modules included with these earphones?',
    'Does this headset come with a 3.5mm jack adapter?',
    'Are custom ear molds provided for these in-ear monitors?',
    'Do these gaming headphones include a detachable voice mic?',
    'Looking for studio headphones with extra-long cords.',
    'I want earphones with changeable decorative plates.',
    'Is a neoprene sleeve part of the headphone package?',
    "Do these kids' headphones come with volume-limiting adapters?",
    'Are sweat-resistant covers available for these workout headphones?',
    'Can I get additional active noise-canceling modules?',
    'Does the purchase include a gold-plated audio connector?'
]

training_data = prepare_training_data(sentences)
print(training_data)
