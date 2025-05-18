import json
import re
import random
import spacy
import string
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from difflib import get_close_matches
from fuzzywuzzy import fuzz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
import csv
import warnings
import requests
import pickle
import numpy as np
import torch
from dateutil import parser
from datetime import datetime, timedelta
from rapidfuzz import process
from textblob import TextBlob
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")
classify = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
def load_medical_keywords(json_file="suppdata/medical_keywords.json"):
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data.get("medical_keywords", [])
medical_keywords = load_medical_keywords()
def extract_keyword(paragraph):
    # Preprocess: Remove punctuation
    paragraph = re.sub(r"[^\w\s]", "", paragraph)  
    doc = nlp(paragraph)
    # Extract candidate keywords (NOUN + ADJ + PROPN)
    keywords = [token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN", "ADJ"}]
    phrases = set()
    for i in range(len(doc) - 1):
        if doc[i].pos_ in {"NOUN", "ADJ"} and doc[i + 1].pos_ == "NOUN":
            phrases.add(f"{doc[i].text.lower()}{doc[i + 1].text.lower()}")  # Merge words
    keywords.extend(phrases)  # Add multi-word phrases to keywords
    if not keywords:
        return None  # Return None if no relevant words found
    # Get embeddings
    paragraph_embedding = model.encode(paragraph, convert_to_tensor=True)
    keyword_embeddings = model.encode(list(keywords), convert_to_tensor=True)

    # Find most similar keyword
    similarity_scores = util.pytorch_cos_sim(paragraph_embedding, keyword_embeddings)[0]
    best_match_index = similarity_scores.argmax().item()

    return list(keywords)[best_match_index]

def load_response_categories(json_file="suppdata/response_categories.json"):
    with open(json_file, "r", encoding="utf-8") as file:
        return json.load(file)

def interpret_response(user_input, json_file="suppdata/response_categories.json"):
    """
    Interpret the user's response and return 'yes', 'no', or 'uncertain'.
    """
    # Load response categories from JSON
    response_data = load_response_categories(json_file)
    positive_responses = set(response_data.get("positive_responses", []))
    negative_responses = set(response_data.get("negative_responses", []))
    uncertain_responses = set(response_data.get("uncertain_responses", []))

    # Standardizing input
    user_input = user_input.lower().strip()
    user_input = user_input.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    
    # Split user input into words for quick lookup
    words = set(user_input.split())
    
    # Direct match check
    if words & positive_responses:
        return "yes"
    if words & negative_responses:
        return "no"
    if words & uncertain_responses:
        return "uncertain"
    
    # Regex for explicit patterns
    pattern_yes = r"(yes\s+i\s+(was|have been|am)\s+[a-z]+|i\s+was\s+suffering|i\s+experienced\s+it)"
    pattern_no = r"(no\s+i\s+(wasn't|was not|am not|haven't been|have not been)\s+[a-z]+|not suffered|i did not suffer|i was not feeling it)"
    
    if re.search(pattern_yes, user_input):
        return "yes"
    if re.search(pattern_no, user_input):
        return "no"

    if any(keyword in user_input for keyword in ["but", "only", "sometimes", "during", "when", "if", "depends", "occasionally"]):
        return "yes"
    
    # Use fuzzy matching for typo detection and correction
    for phrase in positive_responses:
        if fuzz.ratio(user_input, phrase) > 85:
            return "yes"
    for phrase in negative_responses:
        if fuzz.ratio(user_input, phrase) > 85:
            return "no"
    for phrase in uncertain_responses:
        if fuzz.ratio(user_input, phrase) > 85:
            return "uncertain"
    
    # Sentiment analysis for additional context
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(user_input)['compound']
    
    if sentiment_score > 0.3:
        return "yes"
    elif sentiment_score < -0.3:
        return "no"
    
    return "uncertain"

SYMPTOMS_FILE = "Medi AI/symptom_severity.csv"
EMBEDDINGS_FILE = "medical_chatbot_model/symptoms_embeddings.pt"
MODEL_NAME = "all-MiniLM-L6-v2"
def load_word_to_number_mapping(json_file="suppdata/word_to_number.json"):
    with open(json_file, "r", encoding="utf-8") as file:
        return json.load(file)

def words_to_numbers(word, json_file="suppdata/word_to_number.json"):
    word_to_num = load_word_to_number_mapping(json_file)
    return word_to_num.get(word.lower(), None)
def is_medical_query(query):
    # First, check if the query contains any medical keywords
    query_lower = query.lower()
    
    # Check for medical keywords in the query
    for keyword in medical_keywords:
        if keyword in query_lower:
            return True

    # If no keyword matched, use zero-shot classification to classify the query
    candidate_labels = ["medical", "non-medical"]
    result = classify(query, candidate_labels)
    
    # If the result label is 'medical' and the score is above 0.5, classify it as a medical query
    if result['labels'][0] == 'medical' and result['scores'][0] > 0.5:
        return True
    
    return False
def load_mappings(json_file="suppdata/mappings.json"):
    with open(json_file, "r", encoding="utf-8") as file:
        return json.load(file)

def extract_days(user_input, json_file="suppdata/mappings.json"):
    mappings = load_mappings(json_file)
    word_to_num = mappings.get("word_to_number", {})
    weekdays = mappings.get("weekdays", [])
    if user_input.isdigit():
        return int(user_input)
    # Convert word numbers to digits
    for word, num in word_to_num.items():
        user_input = re.sub(rf'\b{word}\b', str(num), user_input, flags=re.IGNORECASE)
    
    # Regular expression to capture durations
    match = re.search(r'(\d+)\s*(day|week|month|year)s?', user_input, re.IGNORECASE)
    if match:
        num = int(match.group(1))
        unit = match.group(2).lower()

        if unit == "week":
            return num * 7
        elif unit == "month":
            return num * 30  # Approximate
        elif unit == "year":
            return num * 365  # Approximate
        return num  # Days directly
    words = user_input.lower().split()
    for word in words:
        if word in weekdays:
            today = datetime.now().weekday()
            day_index = weekdays.index(word)
            delta = (today - day_index) % 7 or 7  # Ensure positive values
            return delta
    try:
        date = parser.parse(user_input, fuzzy=True)
        delta = (datetime.now() - date).days
        return max(delta, 0)
    except (ValueError, OverflowError):
        pass
    return -1 # If no match found
def load_city_mappings(json_file="suppdata/city_mappings.json"):
    with open(json_file, "r", encoding="utf-8") as file:
        return json.load(file)

def extract_city(text, json_file="suppdata/city_mappings.json"):
    mappings = load_city_mappings(json_file)
    location_keywords = mappings.get("location_keywords", [])
    words = text.split()
    # 1. Check for city after a location keyword
    for i, word in enumerate(words):
        if word.lower() in location_keywords and i + 1 < len(words):
            return words[i + 1]  # Pick the next word after keyword
    # 2. Check if any word is a capitalized proper noun (possible city name)
    matches = re.findall(r"\b[A-Z][a-z]*\b", text)
    if matches:
        return matches[0]  # Return the first capitalized word
    return None
def load_symptoms(file_path):
    df = pd.read_csv(file_path)
    return df["Symptom"].str.lower().tolist()
def load_sbert_model():
    return SentenceTransformer(MODEL_NAME)
def precompute_embeddings(model, symptoms):
    embeddings = model.encode(symptoms, convert_to_tensor=True)
    torch.save(embeddings, EMBEDDINGS_FILE)  # Save embeddings for faster reuse
    return embeddings
def load_dataset1(file_path):
    df = pd.read_csv(file_path)
    questions = df["Symptom"].astype(str).tolist()
    answers = df["Description"].astype(str).tolist()
    return questions, answers
def encode_questions(questions, model):
    return model.encode(questions, convert_to_tensor=True)
def find_best_match(user_input, questions, answers, question_embeddings, model):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    best_match_idx = scores.argmax().item()
    return answers[best_match_idx]
def load_cached_embeddings():
    try:
        return torch.load(EMBEDDINGS_FILE, weights_only=True)
    except (FileNotFoundError, RuntimeError):
        return None
file_path1 = "medical_chatbot_model/CLASSIFIER_BERT_CONNECTOR.csv"
model1 = SentenceTransformer("all-MiniLM-L6-v2")
questions, answers = load_dataset1(file_path1)
question_embeddings = encode_questions(questions, model1)

def fuzzy_match(user_input, symptoms, threshold=80, max_matches=2):
    match = process.extractOne(user_input.lower(), symptoms)
    return [match[0]] if match and match[1] >= threshold else []

def sbert_match(user_input, model, symptoms, symptoms_embeddings, max_matches=2):
    user_embedding = model.encode([user_input], convert_to_tensor=True)  # Batch input
    scores = util.pytorch_cos_sim(user_embedding, symptoms_embeddings)[0]
    top_matches = [symptoms[i] for i in scores.argsort(descending=True)[:max_matches]]
    return top_matches

def match_symptoms(user_input, model, symptoms, symptoms_embeddings, max_results=2):
    fuzzy_matches = fuzzy_match(user_input, symptoms, max_matches=max_results)
    sbert_matches = sbert_match(user_input, model, symptoms, symptoms_embeddings, max_matches=max_results)
    return list(set(fuzzy_matches + sbert_matches))[:max_results] or ["No symptoms matched."]
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # Speed optimization
def load_synonyms(filepath="suppdata/synonym.json"):
    with open(filepath, "r") as file:
        return json.load(file)

SYNONYMS = load_synonyms()
symp = load_symptoms(SYMPTOMS_FILE)
model = load_sbert_model()

    # Load cached embeddings if available, otherwise compute and save them
symptoms_embeddings = load_cached_embeddings()
if symptoms_embeddings is None:
    symptoms_embeddings = precompute_embeddings(model, symp)
    
def replace_with_synonyms(text):
    """Replace words in text using a synonym dictionary efficiently."""
    doc = nlp(text)
    replaced_words = [
        random.choice(SYNONYMS.get(token.text.lower(), [token.text])) if token.is_alpha else token.text
        for token in doc
    ]
    return " ".join(replaced_words)

def correct_grammar(sentence):
    """Fix minor grammar issues using TextBlob per sentence."""
    return str(TextBlob(sentence).correct())

def convert_passive_to_active(sentence):
    """Convert passive voice sentences into active voice using SpaCy."""
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ == "nsubjpass":  # Passive subject
            subject = token.text
            verb = next((child for child in token.head.children if child.dep_ == "aux"), token.head).text
            obj = next((child.text for child in token.head.children if child.dep_ == "dobj"), "")
            active_sentence = f"{obj} {verb} {subject}" if obj else f"{subject} {verb}"
            return active_sentence.capitalize()
    return sentence  # Return original if no passive voice found

def restructure_paragraph(text):
    """Enhance fluency by processing each sentence separately."""
    sentences = text.split(". ")  # Split on sentence boundaries
    processed_sentences = []
    
    for sentence in sentences:
        step1 = replace_with_synonyms(sentence)
        step2 = correct_grammar(step1)
        step3 = convert_passive_to_active(step2)
        
        if not step3.endswith(('.', '!', '?')):
            step3 += '.'
        
        processed_sentences.append(step3)

    return " ".join(processed_sentences)

def get_latitude_longitude(city_name):
    api_key = "pk.7f1620ad9ca12db691c4024fee8c613e"  # Replace with your LocationIQ API key
    url = f"https://us1.locationiq.com/v1/search.php?key={api_key}&q={city_name}&format=json"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        if data:
            latitude = float(data[0]["lat"])
            longitude = float(data[0]["lon"])
            return latitude, longitude
        else:
            print(restructure_paragraph(f"Could not find location for {city_name}"))
            return None
    except Exception as e:
        print(restructure_paragraph(f"Error: {e}"))
        return None
import pandas as pd
from geopy.distance import geodesic
def find_nearby_doctors(specialty, user_location, doctor_file, top_n=2):
    try:
        # Load the doctor data
        doctors = pd.read_csv(doctor_file, encoding='ISO-8859-1')
        # Ensure proper formatting and filter doctors based on specialty
        doctors['Specialization'] = doctors['Specialization'].astype(str).str.strip().str.lower()
        specialty = specialty.strip().lower()
        filtered_doctors = doctors[doctors['Specialization'] == specialty]

        if filtered_doctors.empty:
            print(f"No doctors found for the specialty: {specialty}")
            return []

        user_lat, user_lon = user_location  # Unpack user location

        def calculate_distance(row):
            try:
                doctor_lat, doctor_lon = float(row['Latitude']), float(row['Longitude'])
                return geodesic((user_lat, user_lon), (doctor_lat, doctor_lon)).km
            except ValueError:
                return float('inf')  # Assign a high distance if data is missing

        # Compute distances efficiently using apply()
        filtered_doctors['Distance (km)'] = filtered_doctors.apply(calculate_distance, axis=1)

        # Remove invalid entries and sort by distance
        sorted_doctors = filtered_doctors[filtered_doctors['Distance (km)'] != float('inf')].sort_values(by='Distance (km)')

        # Return top N nearby doctors as a list of dictionaries
        return sorted_doctors[['Doctor Name', 'Hospital Name', 'Distance (km)']].head(top_n).to_dict(orient='records')

    except FileNotFoundError:
        print(f"Error: Doctor file '{doctor_file}' not found.")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

# Load datasets with specified encoding to handle UnicodeDecodeError
def load_dataset(file_path):
    try:
        return pd.read_csv(file_path, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        print(restructure_paragraph(f"Error reading file: {file_path}. Trying a different encoding."))
        return pd.read_csv(file_path, encoding='latin-1')

# Load datasets
training = load_dataset('Medi AI/Training.csv')
testing = load_dataset('Medi AI/Testing.csv')

# Preprocess data
cols = training.columns[:-1]  # All columns except the prognosis
x = training[cols]
y = training['prognosis']

# Map labels to numerical values
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Initialize Extra Trees Classifier
clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

# Symptom Dictionaries
severityDictionary = {}
description_list = {}
precautionDictionary = {}

# Symptom encoding
symptoms_dict = {symptom: index for index, symptom in enumerate(x)}

# Functions for loading additional data
def getDescription():
    global description_list
    with open('Medi AI/symptom_Description.csv', encoding='ISO-8859-1') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)  # Skip the header row
        for row in csv_reader:
            if len(row) >= 2:
                description_list[row[0]] = row[1]
def getSeverityDict():
    global severityDictionary
    with open('Medi AI/symptom_severity.csv', encoding='ISO-8859-1') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)  # Skip the header row
        for row in csv_reader:
            if len(row) >= 2:
                try:
                    severityDictionary[row[0]] = int(row[1])
                except ValueError:
                    print(restructure_paragraph(f"Invalid severity value for symptom: {row[0]}"))
def getprecautionDict():
    global precautionDictionary
    with open('Medi AI/symptom_precaution.csv', encoding='ISO-8859-1') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)  # Skip the header row
        for row in csv_reader:
            if len(row) >= 5:
                precautionDictionary[row[0]] = row[1:5]
# Specialist mapping
def load_specialist_mapping(file_path):
    specialist_mapping = {}
    with open(file_path, 'r', encoding='ISO-8859-1') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)  # Skip the header row
        for row in csv_reader:
            if len(row) >= 2:
                specialist_mapping[row[0].strip()] = row[1].strip()
    return specialist_mapping
def is_question(prompt: str) -> bool:
    """Check if the given prompt is a question based on pattern."""
    prompt = prompt.strip()
    # Define a regex pattern for detecting questions
    question_pattern = re.compile(r'^(who|what|when|where|why|how|is|are|do|does|can|could|should|would|will).', re.IGNORECASE)
    return bool(question_pattern.match(prompt))
# Symptom-based secondary prediction
def sec_predict(symptoms_exp):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_exp:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    return clf.predict([input_vector])

def patient_confusion_detector(input_text, json_file="suppdata/confusion_keywords.json"):
    doc = nlp(input_text.lower().strip())
    with open(json_file, "r", encoding="utf-8") as file:
        confusion_keywords = json.load(file).get("suppdata/confusion_keywords", [])
    for phrase in confusion_keywords:
        if phrase in input_text:
            return True
    for token in doc:
        if token.lemma_ in confusion_keywords:
            return True
    return False
# Queue-based symptom filtering
def queue_based_symptom_filtering(user_symptoms, dataset_path):
    # Load and clean the dataset
    dataset = pd.read_csv(dataset_path, encoding='ISO-8859-1')
    dataset_clean = dataset.drop(columns=dataset.columns[0])  # Drop the first column
    dataset_clean = dataset_clean.applymap(lambda x: str(x).strip().lower() if pd.notna(x) else x)
    # Initialize the queue with all tuples containing the first entered symptom
    symptom_queue = []
    matched_rows = dataset_clean[dataset_clean.apply(lambda row: user_symptoms[0] in row.values, axis=1)]
    if matched_rows.empty:
        print(restructure_paragraph("\nNo related symptoms found for the initial input."))
        return user_symptoms
    # Collect all symptoms from the matched rows
    for _, row in matched_rows.iterrows():
        symptom_queue.extend(symptom for symptom in row if pd.notna(symptom) and symptom not in user_symptoms)
    # Calculate the occurrence of each symptom in the matched rows
    symptom_occurrence = {}
    for _, row in matched_rows.iterrows():
        for symptom in row:
            if pd.notna(symptom):
                symptom_occurrence[symptom] = symptom_occurrence.get(symptom, 0) + 1
    # Sort symptoms by their occurrence in descending order
    sorted_symptoms = sorted(symptom_occurrence.items(), key=lambda x: x[1], reverse=True)
    sorted_symptom_list = [symptom for symptom, _ in sorted_symptoms]
    # Rebuild the queue with sorted symptoms
    symptom_queue = [symptom for symptom in sorted_symptom_list if symptom not in user_symptoms]
    # Process the queue iteratively
    while symptom_queue:
        symptom_to_check = symptom_queue.pop(0)  # Get the first symptom in the queue
        if symptom_to_check=="family_history":
            res=get_user_input(f"Is there anyone in your knowledge suffers the same?")
        else:   
            res = get_user_input(f"Are you experiencing {symptom_to_check.replace('_', ' ')}?").strip().lower()
        response=interpret_response(res)
        if response == "yes":
            user_symptoms.append(symptom_to_check)

            # Refine the matched rows to include only those with the confirmed symptoms
            matched_rows = matched_rows[matched_rows.apply(lambda row: symptom_to_check in row.values, axis=1)]

            # Rebuild the occurrence count and queue with sorted symptoms
            symptom_occurrence = {}
            for _, row in matched_rows.iterrows():
                for symptom in row:
                    if pd.notna(symptom):
                        symptom_occurrence[symptom] = symptom_occurrence.get(symptom, 0) + 1
            sorted_symptoms = sorted(symptom_occurrence.items(), key=lambda x: x[1], reverse=True)
            symptom_queue = [symptom for symptom, _ in sorted_symptoms if symptom not in user_symptoms]
        elif response == "no":
            # Exclude rows containing the symptom and rebuild the queue
            matched_rows = matched_rows[~matched_rows.apply(lambda row: symptom_to_check in row.values, axis=1)]
            # Rebuild the occurrence count and queue with sorted symptoms
            symptom_occurrence = {}
            for _, row in matched_rows.iterrows():
                for symptom in row:
                    if pd.notna(symptom):
                        symptom_occurrence[symptom] = symptom_occurrence.get(symptom, 0) + 1
            sorted_symptoms = sorted(symptom_occurrence.items(), key=lambda x: x[1], reverse=True)
            symptom_queue = [symptom for symptom, _ in sorted_symptoms if symptom not in user_symptoms]
        elif response == "uncertain":
            if is_question(res) or patient_confusion_detector(res):
                explanation = find_best_match(f"What is {symptom_to_check}?", questions, answers, question_embeddings, model1)
                print(restructure_paragraph(explanation))
                symptom_queue.insert(0, symptom_to_check)
                    # Re-ask the same symptom after explanation
            else:  
                continue
        # Rebuild the occurrence count and queue with sorted symptoms
        symptom_occurrence = {}
        for _, row in matched_rows.iterrows():
            for symptom in row:
                if pd.notna(symptom):
                    symptom_occurrence[symptom] = symptom_occurrence.get(symptom, 0) + 1
        sorted_symptoms = sorted(symptom_occurrence.items(), key=lambda x: x[1], reverse=True)
        symptom_queue = [symptom for symptom, _ in sorted_symptoms if symptom not in user_symptoms]   
    return user_symptoms
# Calculate condition and recommend specialist
def calc_condition_with_specialist(exp, days, predicted_disease,specialist_mapping):
    severity_sum = sum(severityDictionary.get(item, 0) for item in exp)
    severity_index = (severity_sum * days) / (len(exp) + 1)
    if severity_index > 13:
        print(restructure_paragraph("You should take consultation from a doctor."))
    else:
        print(restructure_paragraph("It might not be serious, but take precautions."))
    # Recommend specialist if severity index exceeds 25
    if severity_index > 25:
        specialist = specialist_mapping.get(predicted_disease, "General Physician")
        print(restructure_paragraph(f"\nRecommended Specialist: {specialist}"))
        user_city = extract_city(get_user_input("\nPlease enter your city to find nearby doctors: ").strip())
        # Get latitude and longitude of the user's city
        user_location = get_latitude_longitude(user_city)
        if user_location:
        # Find nearby doctors for the predicted disease specialty
            specialist = specialist_mapping.get(predicted_disease, "General Physician")
            nearby_doctors = find_nearby_doctors(specialist, user_location, 'Medi AI/kerala_doctors_dataset.csv')
            if nearby_doctors:
                print(restructure_paragraph("\nNearby Doctors:"))
                for doctor in nearby_doctors:
                    print(f"Doctor: {doctor['Doctor Name']}, Hospital: {doctor['Hospital Name']}")
            else:
                print(restructure_paragraph("No nearby doctors found."))
        else:
            print(restructure_paragraph("Couldn't find location for the entered city."))
def get_user_input(prompt):
        """Reads input from the user, ensuring it works in a subprocess."""
        print(restructure_paragraph(prompt))
        return input().strip()
# Main interaction function with doctor recommendation
def tree_to_code_with_dataset_and_specialist(promp,feature_names, dataset_path, specialist_mapping_file):
    symptoms_present = []
    while True:
        # Get user input and process it
        user_input=match_symptoms(promp, model, symp, symptoms_embeddings)
        symptoms = user_input
        # Process each symptom in the input
        for symptom in symptoms:
            if symptom not in symptoms_dict:
                # Suggest similar symptoms if the entered symptom is not found
                similar_symptoms = get_close_matches(symptom, symptoms_dict.keys(), n=5, cutoff=0.6)
                if similar_symptoms:
                    print(restructure_paragraph(f"\nSymptom '{symptom.replace('_', ' ')}' not found."))
                    print(restructure_paragraph("Did you mean one of these symptoms?"))
                    for index, s in enumerate(similar_symptoms, 1):
                        print(restructure_paragraph(f"{index}. {s.replace('_', ' ')}"))
                    try:
                        choice = int(get_user_input("Enter the number corresponding to your symptom, or 0 to skip: ").strip())
                        if 1 <= choice <= len(similar_symptoms):
                            symptom = similar_symptoms[choice - 1]
                        else:
                            print(restructure_paragraph("Skipping this symptom."))
                            continue
                    except ValueError:
                        print(restructure_paragraph("Invalid input. Skipping this symptom."))
                        continue
                else:
                    print(restructure_paragraph(f"No similar symptoms found for '{symptom.replace('_', ' ')}'. Skipping."))
                    continue
            # Add valid symptoms to the list
            symptoms_present.append(symptom)
        cont = "no"
        if cont != 'yes':
            break
    if "mild_fever" in symptoms_present and "high_fever" in symptoms_present:
        symptoms_present.remove("high_fever")
    # Use queue-based refinement for symptoms
    symptoms_present = queue_based_symptom_filtering(symptoms_present, dataset_path)
    days = extract_days(get_user_input("For how many days have you experienced these symptoms? "))
    # Predict disease
    prediction = sec_predict(symptoms_present)
    predicted_disease = le.inverse_transform(prediction)[0]
    print(restructure_paragraph("\nYou may have: ")+str(predicted_disease))
    print(f"Description: {description_list.get(predicted_disease, 'No description available.')}")
    print(restructure_paragraph("\nPrecautions for")+str(predicted_disease)+(":"))
    for i, precaution in enumerate(precautionDictionary.get(predicted_disease, []), 1):
        print(restructure_paragraph(f"{i}. {precaution}"))
    # Load specialist mapping
    specialist_mapping = load_specialist_mapping(specialist_mapping_file)
    # Calculate condition and recommend specialist
    calc_condition_with_specialist(symptoms_present, days, predicted_disease, specialist_mapping)
    # Get the city name from the user for geolocation
    return predicted_disease
# Load dictionaries
getSeverityDict()
getDescription()
getprecautionDict()
# Start interaction
def diagnosis(promp):
    dataset_path = "Medi AI/dataset.csv"  # Replace with your actual dataset path
    specialist_mapping_file = "Medi AI/disease-specialist.csv"  # Replace with your specialist mapping file path
    model, question_embeddings, answer_embeddings, answers, categories, device = load_embeddings_and_model()
    best_answer, score, best_category = get_best_answer(promp, model, answer_embeddings, answers, categories, device)
    print(best_answer)
    print(restructure_paragraph("Its better to take a diagnosis"))
    addtoken=tree_to_code_with_dataset_and_specialist(promp,cols, dataset_path, specialist_mapping_file)
    print(restructure_paragraph("feel free to ask I am here to clear your doubts"))
    chatbot(addtoken)
#  load embedding sanam and model
def load_embeddings_and_model():
    with open('medical_chatbot_model/question_embeddings.pkl', 'rb') as f:
        question_embeddings = pickle.load(f)
    with open('medical_chatbot_model/answer_embeddings.pkl', 'rb') as f:
        answer_embeddings = pickle.load(f)
    with open('medical_chatbot_model/categories.pkl', 'rb') as f:
        categories = pickle.load(f)
    with open('medical_chatbot_model/answers.pkl', 'rb') as f:
        answers = pickle.load(f)
    model = SentenceTransformer('medical_chatbot_model')
    # Move model and embeddings to GPU (only bengu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    answer_embeddings = torch.tensor(answer_embeddings).to(device)
    model = model.to(device)
    return model, question_embeddings, answer_embeddings, answers, categories, device
# adjustment mariyadhakk veran
def format_answer(answer):
    # Split the answer into paragraphs
    paragraphs = answer.split("\n\n")
    # Add indentation and clean up
    formatted_paragraphs = []
    for paragraph in paragraphs:
        lines = paragraph.strip().split("\n")
        formatted_paragraph = "\n    ".join(lines)
        formatted_paragraphs.append(f"    {formatted_paragraph}")
    # Combine all paragraphs with spacing
    formatted_answer = "\n\n".join(formatted_paragraphs)
    return formatted_answer
def modify_prompt(prompt: str, token: str = None) -> str:
    if token is None:
        return prompt  # Return the original prompt if no replacement token is provided
    return(str(prompt+" "+token))
# Function revent sanam accuracy kootan
def get_best_answer(query, model, answer_embeddings, answers, categories, device):
    # Embed the query using SBERT
    query_embedding = model.encode(query, convert_to_tensor=True).to(device)
    # Compute cosine quality nokkan eth 
    cosine_scores = util.pytorch_cos_sim(query_embedding, answer_embeddings)[0]
    # Get the index of the most similar answer
    best_match_idx = np.argmax(cosine_scores.cpu().numpy())  # Convert to NumPy for indexing
    # Format the best answer
    formatted_answer = format_answer(answers[best_match_idx])
    # Return the best matching answer
    return formatted_answer, cosine_scores[best_match_idx].item(), categories[best_match_idx]
#interactive chatbot loop
def chatbot(query=None):
    # Load saved data (embeddings and model)
    model, question_embeddings, answer_embeddings, answers, categories, device = load_embeddings_and_model()
    addtok=None
    done=0
    while True:
        if query is None:
            query = input()
        if query.lower() == 'exit':
            print(restructure_paragraph("Good bye!"))
            break
        query=modify_prompt(query,addtok)
        if is_medical_query(query)==False:
            print(restructure_paragraph("See i am a virtual doctor (AI) please provide me a detailed medical query for your solution"))
            query=None
            continue
        # Get the best answer to the query
        best_answer, score, best_category = get_best_answer(query, model, answer_embeddings, answers, categories, device)
        if done>4 or done==0:
            addtok=extract_keyword(best_answer)
            done=0
        done+=1
        # Decode the category back to a string
        label_encoder = LabelEncoder()
        label_encoder.fit(categories)  # Ensure label encoder is properly initialized
        best_category_label = label_encoder.inverse_transform([best_category])[0]

        # Display the formatted answer
        print(restructure_paragraph(best_answer))
        query=None
def determine_user_intent():
    # Load keywords
    with open("suppdata/intent_keywords.json", "r") as file:
        keywords = json.load(file)
    diagnosis_keywords = set(keywords["diagnosis_keywords"])
    qa_keywords = set(keywords["qa_keywords"])
    print(restructure_paragraph("Hello! I am a virtual doctor. How can I assist you?"))
    while True:
        user_input = input("").strip().lower()
        if user_input == "exit":
            print(restructure_paragraph("Goodbye!"))
            break
        if is_medical_query(user_input)==False:
            print(restructure_paragraph("See i am a virtual doctor please provide me a detailed medical query for your solution"))
            continue
        # Preprocess input
        user_words = re.findall(r"\b\w+\b", user_input)
        # Calculate intent scores using exact & fuzzy matching
        diagnosis_score = sum(1 for word in user_words if word in diagnosis_keywords) + \
                          sum(0.5 for word in user_words if get_close_matches(word, diagnosis_keywords, cutoff=0.8))
        qa_score = sum(1 for word in user_words if word in qa_keywords) + \
                   sum(0.5 for word in user_words if get_close_matches(word, qa_keywords, cutoff=0.8))
        # Determine intent
        if diagnosis_score > qa_score:
            diagnosis(user_input)
            break
        elif qa_score > diagnosis_score:
            
            chatbot(user_input)
            break
        else:
            print(restructure_paragraph("I'm not sure what you mean. its better you explain your issue towards me clearly?"))
# Run the function
determine_user_intent()
