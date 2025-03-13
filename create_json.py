import json

intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "Good day", "Greetings", "What's up?", "How's it going?"],
            "responses": [
                "Hey there! I'm your farming assistant. What's on your mind today?",
                "Hello! Ready to talk farming? What do you need help with?",
                "Hi! Let's solve your farming problems together. What's up?"
            ],
            "context": []
        },
        {
            "tag": "crops",
            "patterns": [
                "What crops should I plant?",
                "Best crops for this season",
                "How to grow crops",
                "Planting advice",
                "Crop recommendations",
                "When to harvest",
                "Crop rotation"
            ],
            "responses": [
                "Hmm, let's figure out the best crops for you. What's your location and the current season?",
                "Crop selection depends on soil, climate, and market demand. What's your soil type?",
                "Crop rotation is key! What did you grow last season? I'll help you plan the next."
            ],
            "context": ["crop_advice"]
        },
        {
            "tag": "soil",
            "patterns": [
                "How is my soil quality?",
                "Soil testing",
                "Soil improvement",
                "Soil pH",
                "Soil fertility",
                "Composting tips"
            ],
            "responses": [
                "Soil health is crucial! Have you tested your soil's pH and nutrient levels?",
                "Improving soil? Try composting, cover crops, or organic fertilizers. What's your soil type?",
                "Let's talk about your soil. What specific issues are you facing?"
            ],
            "context": ["soil_advice"]
        }
    ]
}

# Write the JSON file
with open('agri_intents.json', 'w', encoding='utf-8') as f:
    json.dump(intents, f, indent=4, ensure_ascii=False)
print("JSON file created successfully!")
