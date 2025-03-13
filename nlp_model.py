import json
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import os

class ChatBot:
    def __init__(self):
        print("Initializing ChatBot...")
        self.lemmatizer = WordNetLemmatizer()
        
        if not os.path.exists('agri_intents.json'):
            raise FileNotFoundError("agri_intents.json not found in current directory")
            
        try:
            with open('agri_intents.json', 'r', encoding='utf-8-sig') as file:
                self.intents = json.load(file)
                print("Successfully loaded intents file")
        except json.JSONDecodeError as e:
            print(f"Error reading JSON file: {e}")
            raise
            
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_letters = ['?', '!', '.', ',']

    def preprocess_data(self):
        print("Starting data preprocessing...")
        # Clear existing data
        self.words = []
        self.classes = []
        self.documents = []

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern)
                self.words.extend(word_list)
                self.documents.append((word_list, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(word.lower()) for word in self.words if word not in self.ignore_letters]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

        print(f"Preprocessed {len(self.documents)} documents")
        print(f"Found {len(self.classes)} classes: {self.classes}")
        print(f"Found {len(self.words)} unique lemmatized words")

        # Save words and classes to files
        pickle.dump(self.words, open('words.pkl', 'wb'))
        pickle.dump(self.classes, open('classes.pkl', 'wb'))
        print("Saved words.pkl and classes.pkl")

        return self.words, self.classes, self.documents

    def create_training_data(self):
        print("Creating training data...")
        training = []
        output_empty = [0] * len(self.classes)

        for document in self.documents:
            bag = []
            word_patterns = document[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]

            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(document[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training, dtype=object)

        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        print(f"Created {len(train_x)} training samples")
        return np.array(train_x), np.array(train_y)

    def build_model(self):
        print("Building neural network model...")
        model = Sequential()
        model.add(Dense(128, input_shape=(len(self.words),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.classes), activation='softmax'))

        sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
        print("Model built successfully")
        return model

    def train(self):
        print("\n=== Starting Training Process ===\n")
        
        # Preprocess data
        print("Step 1: Preprocessing data...")
        self.preprocess_data()
        
        # Create training data
        print("\nStep 2: Creating training data...")
        train_x, train_y = self.create_training_data()

        # Build and train the model
        print("\nStep 3: Building and training model...")
        model = self.build_model()
        
        print("\nTraining the model (this may take a few minutes)...")
        hist = model.fit(np.array(train_x), np.array(train_y), 
                        epochs=200, 
                        batch_size=5, 
                        verbose=1)
        
        # Save the model
        model.save('chatbot_model.h5')
        print("\nModel trained and saved as 'chatbot_model.h5'")
        print("\n=== Training Complete ===")

def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    from tensorflow.keras.models import load_model
    model = load_model('chatbot_model.h5')
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    
    p = bag_of_words(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure I understand. Could you please rephrase that?"
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

if __name__ == "__main__":
    try:
        print("Starting AgriBot training process...")
        chatbot = ChatBot()
        chatbot.train()
        print("\nTraining completed successfully! The model is ready to chat!")
        
    except Exception as e:
        print(f"\nError occurred during training: {str(e)}")
        print("Please check the error message above and ensure all files are in place.")
