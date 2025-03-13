from flask import Flask, render_template, request, jsonify, session
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
import os
from datetime import datetime
import spacy
import requests

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")  # Use a more advanced NLP model

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session management

def get_weather(city):
    API_KEY = "98ee7b853ed091b07258e3a98344aa28"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    
    if data["cod"] != "404":
        return f"The temperature in {city} is {data['main']['temp']}°C with {data['weather'][0]['description']}."
    else:
        return "I couldn't fetch the weather details. Please check the city name."

class AgriBot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
        # Check if required files exist
        required_files = ['chatbot_model.h5', 'words.pkl', 'classes.pkl', 'agri_intents.json']
        for file in required_files:
            if not os.path.exists(file):
                print(f"ERROR: {file} not found!")
                raise FileNotFoundError(f"Required file {file} not found!")
        
        print("Loading model and data...")
        try:
            self.model = load_model('chatbot_model.h5')
            self.words = pickle.load(open('words.pkl', 'rb'))
            self.classes = pickle.load(open('classes.pkl', 'rb'))
            self.intents = json.load(open('agri_intents.json', 'r', encoding='utf-8-sig'))
            self.context = {}
            print("All components loaded successfully")
        except Exception as e:
            print(f"Error loading files: {str(e)}")
            raise

    def clean_up_sentence(self, sentence):
        doc = nlp(sentence)
        return [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(self, intents_list, user_id):
        tag = intents_list[0]['intent'] if intents_list else "unknown"
        
        if tag == "weather_advice":
            city = "Delhi"  # Change this to dynamically fetch from user input
            return get_weather(city)

        # Get user context
        user_context = self.context.get(user_id, {})
        current_time = datetime.now()
        
        # Update context with new interaction
        if user_id not in self.context:
            self.context[user_id] = {
                'last_intent': tag,
                'interaction_count': 1,
                'last_interaction': current_time
            }
        else:
            self.context[user_id]['last_intent'] = tag
            self.context[user_id]['interaction_count'] += 1
            self.context[user_id]['last_interaction'] = current_time

        # Fallback to predefined responses
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                return np.random.choice(intent['responses'])
        
        return "I'm not sure how to respond to that. Can you provide more details?"

# Initialize the bot
try:
    print("Initializing bot...")
    bot = AgriBot()
    print("Bot initialized successfully!")
except Exception as e:
    print(f"Error initializing bot: {str(e)}")
    bot = None

@app.route('/')
def home():
    if 'user_id' not in session:
        session['user_id'] = str(datetime.now().timestamp())
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    if bot is None:
        return jsonify({'response': 'Sorry, the bot is not properly initialized.'})
    
    try:
        user_message = request.json['message']
        user_id = session.get('user_id', 'default_user')
        
        # Get response
        ints = bot.predict_class(user_message)
        response = bot.get_response(ints, user_id)
        
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error processing message: {str(e)}")
        return jsonify({'response': 'Sorry, there was an error processing your message.'})

if __name__ == '__main__':
    print("Starting AgriBot Web Interface...")
    app.run(debug=True)
