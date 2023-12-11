import json

# Load the updated dialogues
with open('../conversationalPairs/dialoguePairOne.json', 'r') as file:
    dialogue_data = json.load(file)

# Define mappings for various preferences
brand_mapping = {
    "Sony": 0,
    "Skull-candy": 1,
    "Bose": 2,
    "JBL": 3,
    "Sennheiser": 4,
    "Panasonic": 5,
    "Beats": 6,
    "Skullcandy": 7,
    "Samsung": 8,
    "Harman Kardon": 9,
    "Audio-Technica": 10,
    "unknown": 11
}

color_mapping = {
    "black": 0,
    "purple": 1,
    "red": 2,
    "gold": 3,
    "green": 4,
    "blue": 5,
    "pink": 6,
    "silver": 7,
    "white": 8,
    "orange": 9,
    "yellow": 10,
    "turquoise": 11,
    "vibrant": 12,
    "pastel": 13,
    "neon": 14,
    "matte black": 15,
    "electric blue": 16,
    "matte silver": 17,
    "pearl white": 18,
    "charcoal grey": 19,
    "sunburst yellow": 20,
    "matte grey": 21,
    "neon orange": 22,
    "lime green": 23,
    "burgundy": 24,
    "vibrant orange": 25,
    "glossy black": 26,
    "vibrant turquoise": 27,
    "rose gold": 28,
    "coral": 29,
    "bright red": 30,
    "classic black": 31,
    "electric yellow": 32,
    "unknown": 33
}
battery_life_mapping = {
    "all-day": 0,
    "12 hours": 1,
    ">14 hours": 2,
    "long": 3,
    "decent": 4,
    "good": 5,
    ">10Hours": 6,
    ">15Hours": 7,
    "full day": 8,
    ">8Hours": 9,
    ">12Hours": 10,
    "extended": 11,
    "excellent": 12,
    "great": 13,
    "10-20Hours": 14,
    "12-24Hours": 15,
    "around 18Hours": 16,
    "around 20Hours": 17,
    "ultra-long": 18,
    "around 15Hours": 19,
    "around 16Hours": 20,
    "around 12Hours": 21,
    "around 14Hours": 22,
    "up to 24Hours": 23,
    "unknown": 24
}
price_range_mapping = {
    "<20$": 0,
    "20$-50$": 1,
    "50$-100$": 2,
    ">100$": 3,
    "affordable": 4,
    "budget-friendly": 5,
    "around 70$": 6,
    "around 50$": 7,
    "varied": 8,
    "<40$": 9,
    "<60$": 10,
    "<75$": 11,
    "<100$": 12,
    "<30$": 13,
    "around 150$": 14,
    "mid-range": 15,
    "around 120$": 16,
    "luxury": 17,
    "around 60$": 18,
    "<250$": 19,
    "around 200$": 20,
    "high-end": 21,
    "unknown": 22
}


# Function to handle null or missing values
def handle_null(value, mapping):
    return mapping.get(value, mapping["unknown"])

# Initialize the contextual data structure
contextual_data = {
    "brand_preference": [],
    "color_preference": [],
    "price_range_preference": [],
    "battery_life_preference": []
}

# Process each dialogue pair to extract and encode context
for dialogue in dialogue_data:
    context = dialogue["context"]
    
    # Encode brand preference
    contextual_data["brand_preference"].append(handle_null(context.get("BrandInquiry"), brand_mapping))
    
    # Encode color preference
    contextual_data["color_preference"].append(handle_null(context.get("ColorInquiry"), color_mapping))
    
    # Encode battery life preference
    contextual_data["battery_life_preference"].append(handle_null(context.get("BatteryLifeInquiry"), battery_life_mapping))
    
    # Encode price range preference
    contextual_data["price_range_preference"].append(handle_null(context.get("PriceRangeInquiry"), price_range_mapping))

# Save the processed contextual data
with open('contextual_data.json', 'w') as f:
    json.dump(contextual_data, f, indent=4)
