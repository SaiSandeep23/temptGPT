import spacy
from spacy.training import Example
import random
from data.raw.ACCESSORIES import ACCESSORIES
from data.raw.AVAILABILITY import AVAILABILITY
from data.raw.BATTERY_LIFE import BATTERY_LIFE
from data.raw.BRAND_NAME import BRAND_NAME
from data.raw.COLOR import COLOR
from data.raw.CONNECTIVITY import CONNECTIVITY
from data.raw.HEADPHONE_TYPE import HEADPHONE_TYPE
from data.raw.MATERIAL import MATERIAL
from data.raw.MICROPHONE import MICROPHONE
from data.raw.NOISE_CANCELLING import NOISE_CANCELLING
from data.raw.PRICE_RANGE import PRICE_RANGE
from data.raw.SOUND_QUALITY import SOUND_QUALITY
from data.raw.USER_RATING import USER_RATING
from data.raw.WARRANTY import WARRANTY
from data.raw.WATER_RESISTANT import WATER_RESISTANT
from itertools import chain
import os
from spacy.symbols import ORTH
def to_lower_case(text, annotations):
    lower_text = text.lower()
    new_entities = []
    for start, end, label in annotations['entities']:
        new_entities.append((start, end, label))
    return lower_text, {'entities': new_entities}

# Load a blank model
nlp = spacy.blank("en")

# Create a NER pipeline
if "ner" not in nlp.pipe_names:
    ner = nlp.create_pipe("ner")
    nlp.add_pipe("ner")

# Add new entity labels to the NER pipeline
# labels = [
#     "HEADPHONE_TYPE", "BRAND_NAME", "PRICE_RANGE", "CONNECTIVITY",
#     "BATTERY_LIFE", "NOISE_CANCELLING", "WATER_RESISTANT", "COLOR",
#     "MATERIAL", "SOUND_QUALITY", "MICROPHONE", "WARRANTY",
#     "AVAILABILITY", "USER_RATING", "ACCESSORIES"
# ]

labels = [
    "BRAND_NAME", "BATTERY_LIFE", "COLOR","PRICE_RANGE"
]

for label in labels:
    ner.add_label(label)

# Prepare training data
# TRAIN_DATA = list(chain(
# HEADPHONE_TYPE + BRAND_NAME + PRICE_RANGE + CONNECTIVITY,
#     BATTERY_LIFE + NOISE_CANCELLING + WATER_RESISTANT + COLOR,
#     MATERIAL + SOUND_QUALITY + MICROPHONE + WARRANTY,
#     AVAILABILITY + USER_RATING + ACCESSORIES
# )) 
TRAIN_DATA = [
    ("Do you stock any Sony headphones?", {'entities': [(17, 21, 'BRAND_NAME')]}),
    ("Do you have these headphones in black?", {'entities': [(32, 37, 'COLOR')]}),
    ("Do these earbuds offer all-day battery life?", {'entities': [(22, 36, 'BATTERY_LIFE')]}),
    ("No, I need headphones which have a battery life more than 14 hours on full charge", {'entities': [(33, 60, 'BATTERY_LIFE')]}),
    ("Ho great then, Can I find Skullcandy earbuds in your store?", {'entities': [(23, 33, 'BRAND_NAME')]}),
    ("I need under 20$ and purple color", {'entities': [(7, 16, 'PRICE_RANGE'), (21, 27, 'COLOR')]}),

    ("Can I get a pair of earbuds with a long battery life?", {'entities': [(32, 44, 'BATTERY_LIFE')]}),
    ("I'm looking for Skullcandy headphones, do you carry them?", {'entities': [(15, 25, 'BRAND_NAME')]}),
    ("Is there any headphone available in red color for under 50$?", {'entities': [(36, 39, 'COLOR'), (44, 51, 'PRICE_RANGE')]}),
    
    ("What's the battery duration for these wireless earphones?", {'entities': [(14, 29, 'BATTERY_LIFE')]}),
    ("Can you tell me how long the earbuds last on one charge?", {'entities': [(28, 42, 'BATTERY_LIFE')]}),
    ("How long does the battery last without charging?", {'entities': [(18, 34, 'BATTERY_LIFE')]}),
    ("What is the expected battery life for these earphones on a single charge?", {'entities': [(23, 50, 'BATTERY_LIFE')]}),
    
    ("Do you have earbuds from Skullcandy available?", {'entities': [(20, 30, 'BRAND_NAME')]}),
    ("Which Skullcandy models do you carry in stock?", {'entities': [(7, 17, 'BRAND_NAME')]}),
    ("I'm looking for earphones, preferably Skullcandy. What do you have?", {'entities': [(32, 42, 'BRAND_NAME')]}),
    ("Are there any Skullcandy headphones on sale right now?", {'entities': [(12, 22, 'BRAND_NAME')]}),
    
    ("I'd like a pair of headphones for less than 50 dollars in a blue shade.", {'entities': [(31, 48, 'PRICE_RANGE'), (52, 56, 'COLOR')]}),
    ("Could I find headphones in the color green for about 30$?", {'entities': [(34, 39, 'COLOR'), (48, 53, 'PRICE_RANGE')]}),
    ("What options are available for pink earbuds under 25 dollars?", {'entities': [(29, 33, 'COLOR'), (40, 57, 'PRICE_RANGE')]}),
    ("I'm looking for navy-colored headphones within a 100-dollar budget.", {'entities': [(16, 30, 'COLOR'), (45, 58, 'PRICE_RANGE')]}),
    ("Do you offer any earphones in red that are priced below 20 dollars?", {'entities': [(26, 29, 'COLOR'), (51, 67, 'PRICE_RANGE')]}),
    
    ("These headphones have a battery life of up to 15 hours, right?", {'entities': [(25, 45, 'BATTERY_LIFE')]}),
    ("What's the battery duration for the wireless earbuds?", {'entities': [(14, 30, 'BATTERY_LIFE')]}),
    ("Can you tell me the battery life expectancy for these models?", {'entities': [(23, 42, 'BATTERY_LIFE')]}),
    ("Are these headphones equipped with extended battery life?", {'entities': [(34, 49, 'BATTERY_LIFE')]}),
    
    ("Is the Skullcandy model available in this store?", {'entities': [(7, 17, 'BRAND_NAME')]}),
    ("Do you have Skullcandy products in stock?", {'entities': [(12, 22, 'BRAND_NAME')]}),
    ("I'm interested in the latest Skullcandy earphones. Do you have them?", {'entities': [(25, 35, 'BRAND_NAME')]}),
    ("Which Skullcandy headphones do you recommend?", {'entities': [(7, 17, 'BRAND_NAME')]}),
    
    ("Do your headphones support extended battery life for continuous use?", {'entities': [(26, 50, 'BATTERY_LIFE')]}),
    ("Can these headphones provide battery life for an entire day?", {'entities': [(27, 52, 'BATTERY_LIFE')]}),
    ("I'm interested in earbuds with long-lasting battery life. Any suggestions?", {'entities': [(31, 52, 'BATTERY_LIFE')]}),
    ("How long is the battery life for these earbuds on a single charge?", {'entities': [(23, 51, 'BATTERY_LIFE')]}),
    ("Are these wireless earbuds designed for all-day use without recharging?", {'entities': [(36, 59, 'BATTERY_LIFE')]}),
    ("What is the typical battery life span for these Bluetooth earbuds?", {'entities': [(25, 49, 'BATTERY_LIFE')]}),
    ("Can I expect these earbuds to last a full day on one battery charge?", {'entities': [(30, 59, 'BATTERY_LIFE')]}),
    ("Do these earphones offer a battery life that can endure long hours?", {'entities': [(25, 58, 'BATTERY_LIFE')]}),
    
    ("I need earbuds with an all-day battery capacity for my trips.", {'entities': [(19, 37, 'BATTERY_LIFE')]}),
    ("Do you have any headphones that offer all-day usage on a single charge?", {'entities': [(38, 61, 'BATTERY_LIFE')]}),
    ("Looking for earphones that can last all-day on a full charge.", {'entities': [(36, 56, 'BATTERY_LIFE')]}),
    ("Are these models capable of all-day playback without needing to recharge?", {'entities': [(27, 57, 'BATTERY_LIFE')]}),
    ("I want earbuds that won’t run out quickly, maybe something with all-day battery life.", {'entities': [(61, 75, 'BATTERY_LIFE')]}),
    ("What are the options for earbuds with all-day power endurance?", {'entities': [(37, 53, 'BATTERY_LIFE')]}),
    ("Can these earbuds really last all-day on a single battery cycle?", {'entities': [(31, 56, 'BATTERY_LIFE')]}),
    ("Looking for all-day lasting earphones within my budget.", {'entities': [(12, 29, 'BATTERY_LIFE')]}),
    ("Are these earbuds equipped with an all-day battery?", {'entities': [(34, 45, 'BATTERY_LIFE')]}),
    ("I'm looking for earbuds that can last all-day without recharging.", {'entities': [(42, 63, 'BATTERY_LIFE')]}),
    ("What is the battery life like, can it support all-day use?", {'entities': [(48, 56, 'BATTERY_LIFE')]}),
    ("Do you offer any earphones that promise all-day listening?", {'entities': [(41, 59, 'BATTERY_LIFE')]}),
    ("I need earbuds that won't need charging all day, any suggestions?", {'entities': [(38, 54, 'BATTERY_LIFE')]}),
    ("How long will these last? I’m looking for something with all-day capacity.", {'entities': [(56, 69, 'BATTERY_LIFE')]}),
    ("Can I expect these headphones to provide all-day audio playback?", {'entities': [(44, 59, 'BATTERY_LIFE')]}),
    ("Is it true that these earbuds have all-day performance?", {'entities': [(37, 53, 'BATTERY_LIFE')]}),
    ("These earbuds are advertised with all-day battery, is that correct?", {'entities': [(35, 47, 'BATTERY_LIFE')]}),
    ("Can I use these earbuds for all-day listening without the battery dying?", {'entities': [(25, 48, 'BATTERY_LIFE')]}),
    ("What's the battery life expectancy for these earbuds? Is it all-day?", {'entities': [(48, 55, 'BATTERY_LIFE')]}),
    ("I heard these earbuds have impressive battery life, like all-day duration?", {'entities': [(53, 64, 'BATTERY_LIFE')]}),
    ("Are you saying these earbuds won’t need a battery charge for an entire day?", {'entities': [(43, 69, 'BATTERY_LIFE')]}),
    ("Do these earbuds support day-long battery life for extensive usage?", {'entities': [(25, 44, 'BATTERY_LIFE')]}),
    ("How many hours can these earbuds last? I need something with an all-day battery.", {'entities': [(59, 69, 'BATTERY_LIFE')]}),
    ("Is it true that the battery of these earbuds lasts all day on a single charge?", {'entities': [(39, 62, 'BATTERY_LIFE')]}),
    ("The earbuds boast an all-day battery life, perfect for long commutes.", {'entities': [(20, 34, 'BATTERY_LIFE')]}),
    ("I'm looking for headphones with all-day usage capability.", {'entities': [(34, 42, 'BATTERY_LIFE')]}),
    ("Are the earbuds with all-day battery life available in your store?", {'entities': [(18, 36, 'BATTERY_LIFE')]}),
    ("I prefer headphones that guarantee all-day battery life for travel.", {'entities': [(31, 49, 'BATTERY_LIFE')]}),
    ("Looking for a model that offers all-day battery life; do you have any?", {'entities': [(28, 46, 'BATTERY_LIFE')]}),
    ("What options do you have for earbuds featuring all-day battery life?", {'entities': [(42, 60, 'BATTERY_LIFE')]}),
    ("Is there a version of these headphones that supports all-day battery life?", {'entities': [(47, 65, 'BATTERY_LIFE')]}),
    ("Can you recommend wireless earphones with all-day battery life?", {'entities': [(38, 56, 'BATTERY_LIFE')]}),
    ("Do any of your earbuds come with all-day battery life as a feature?", {'entities': [(30, 48, 'BATTERY_LIFE')]}),
    ("I'm interested in earphones that provide all-day battery life. Any suggestions?", {'entities': [(39, 57, 'BATTERY_LIFE')]}),
    ("How effective is the all-day battery life on these new earbuds?", {'entities': [(23, 41, 'BATTERY_LIFE')]}),
    ("Are these sports headphones equipped with all-day battery life?", {'entities': [(36, 54, 'BATTERY_LIFE')]}),
    
    ("Are these headphones capable of lasting a full day on one charge?", {'entities': [(31, 60, 'BATTERY_LIFE')]}),
    ("What's the maximum battery life I can expect from these earbuds?", {'entities': [(28, 53, 'BATTERY_LIFE')]}),
    ("Can these wireless headphones sustain a whole day's use on a single charge?", {'entities': [(34, 67, 'BATTERY_LIFE')]}),
    ("How many hours of playback do these earbuds offer with one battery cycle?", {'entities': [(32, 66, 'BATTERY_LIFE')]}),
    ("Do these earphones provide enough battery life to last through an entire workday?", {'entities': [(28, 66, 'BATTERY_LIFE')]}),
    ("Is the battery life on these earbuds sufficient for continuous use throughout the day?", {'entities': [(23, 69, 'BATTERY_LIFE')]}),
    ("Will the battery of these earbuds last for an entire day of usage?", {'entities': [(28, 57, 'BATTERY_LIFE')]}),
    ("Could you tell me about the battery endurance of these earbuds for daily use?", {'entities': [(31, 62, 'BATTERY_LIFE')]}),
    ("Are these earbuds designed for prolonged use with their battery capacity?", {'entities': [(39, 65, 'BATTERY_LIFE')]}),
    ("What is the expected battery runtime for these earbuds in regular use?", {'entities': [(27, 60, 'BATTERY_LIFE')]}),
    
    ("Are there any new releases from Audio-Technica?", {"entities": [(29, 43, "BRAND_NAME")]}),
    ("I'm interested in JBL's latest headphones.", {"entities": [(16, 19, "BRAND_NAME")]}),
    ("Do you sell Philips over-ear headphones?", {"entities": [(11, 18, "BRAND_NAME")]}),
    ("Looking for AKG's noise-cancelling headphones.", {"entities": [(14, 17, "BRAND_NAME")]}),
    ("I want to check out the latest from Panasonic.", {"entities": [(33, 42, "BRAND_NAME")]}),
    ("Are Marshall headphones part of your inventory?", {"entities": [(4, 12, "BRAND_NAME")]}),
    ("Is there a discount on Bang & Olufsen products?", {"entities": [(20, 35, "BRAND_NAME")]}),
    ("Do you carry Grado's latest headphone collection?", {"entities": [(12, 17, "BRAND_NAME")]}),
    ("I'm searching for Logitech gaming headsets.", {"entities": [(17, 25, "BRAND_NAME")]}),
    ("Are Harman Kardon earphones good for daily use?", {"entities": [(4, 18, "BRAND_NAME")]}),
    
    ("Do you stock the rose gold variant of these earphones?", {"entities": [(16, 25, "COLOR")]}),
    ("I'm looking for headphones in a matte gray finish.", {"entities": [(29, 39, "COLOR")]}),
    ("Is the green model of these headphones available?", {"entities": [(7, 12, "COLOR")]}),
    ("Are there any headphones in a vibrant yellow?", {"entities": [(31, 44, "COLOR")]}),
    ("Do these come in a sleek silver color?", {"entities": [(19, 31, "COLOR")]}),
    ("I want headphones in a cool teal shade.", {"entities": [(25, 34, "COLOR")]}),
    ("Is there a purple option for these headphones?", {"entities": [(12, 18, "COLOR")]}),
    ("Do you have headphones in a bright orange color?", {"entities": [(27, 39, "COLOR")]}),
    ("I'm interested in earbuds with a pink hue.", {"entities": [(30, 34, "COLOR")]}),
    ("Can you show me headphones in a metallic bronze?", {"entities": [(34, 48, "COLOR")]}),
    ("Are these headphones offered in a navy blue color?", {"entities": [(36, 45, "COLOR")]}),
    ("Do these earphones come in a deep maroon?", {"entities": [(29, 40, "COLOR")]}),
    ("I want to see the headphones in an ivory tone.", {"entities": [(31, 36, "COLOR")]}),
    ("Is there a charcoal option available for these headphones?", {"entities": [(12, 20, "COLOR")]}),
    
    ("What is the average playtime on a single battery charge for these headphones?", {"entities": [(12, 52, "BATTERY_LIFE")]}),
    ("How many hours of usage can I get from these headphones' battery?", {"entities": [(9, 57, "BATTERY_LIFE")]}),
    ("Do these headphones have a quick charge feature for the battery?", {"entities": [(25, 52, "BATTERY_LIFE")]}),
    ("Is the battery life sufficient for long flights on these headphones?", {"entities": [(9, 43, "BATTERY_LIFE")]}),
    ("Tell me about the battery endurance of these gaming headphones.", {"entities": [(16, 32, "BATTERY_LIFE")]}),
    ("Can I expect a full day's battery life from these Bluetooth headphones?", {"entities": [(12, 32, "BATTERY_LIFE")]}),
    ("What is the charging time to full battery on these headphones?", {"entities": [(12, 36, "BATTERY_LIFE")]}),
    ("How frequently do I need to charge the battery of these headphones?", {"entities": [(23, 53, "BATTERY_LIFE")]}),
    ("Do these headphones feature a battery-saving mode?", {"entities": [(27, 43, "BATTERY_LIFE")]}),
    ("Are the batteries replaceable in these wireless headphones?", {"entities": [(7, 23, "BATTERY_LIFE")]}),
    ("What's the standby time for the battery in these headphones?", {"entities": [(13, 35, "BATTERY_LIFE")]}),
    ("How does the battery perform during cold weather in these headphones?", {"entities": [(16, 44, "BATTERY_LIFE")]}),
    ("Is there a low battery indicator on these headphones?", {"entities": [(10, 28, "BATTERY_LIFE")]}),
    ("Can the battery in these headphones last through a long workday?", {"entities": [(16, 55, "BATTERY_LIFE")]}),
    ("What is the battery capacity of these headphones in mAh?", {"entities": [(16, 47, "BATTERY_LIFE")]}),
    ("Do you have any headphones with a battery life of over 10 hours?", {"entities": [(26, 50, "BATTERY_LIFE")]}),
    ("What is the longest battery life available for your wireless models?", {"entities": [(12, 58, "BATTERY_LIFE")]}),
    ("Are these headphones equipped with a fast charging battery system?", {"entities": [(34, 57, "BATTERY_LIFE")]}),
    ("How effective is the battery life in high-use scenarios for these headphones?", {"entities": [(23, 47, "BATTERY_LIFE")]}),
    ("Can you tell me the expected battery lifespan for these earphones?", {"entities": [(23, 43, "BATTERY_LIFE")]}),
    ("What's the maximum battery capacity for these Bluetooth headphones?", {"entities": [(12, 34, "BATTERY_LIFE")]}),
    ("Are there any battery usage indicators on these headphones?", {"entities": [(13, 34, "BATTERY_LIFE")]}),
    ("How long can these headphones operate on a single charge?", {"entities": [(25, 47, "BATTERY_LIFE")]}),

]
# print(TRAIN_DATA)

special_case = [{ORTH: "all-day"}]
nlp.tokenizer.add_special_case("all-day", special_case)

TRAIN_DATA = [to_lower_case(text, annotations) for text, annotations in TRAIN_DATA]


# Train the model
optimizer = nlp.begin_training()
for itn in range(120):  # Number of iterations
    random.shuffle(TRAIN_DATA)
    losses = {}
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.3, losses=losses)
    print(f"Iteration {itn} losses: {losses}")

# Save the trained model
model_dir = "models/trained"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
nlp.to_disk("models/trained/entity_recog_model")
