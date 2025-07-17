import os
import re
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import speech_recognition as sr
import sounddevice # Or import pyaudio
import time
import nltk # Needed if clean_text uses it heavily, otherwise maybe optional

# --- Configuration ---
MODEL_PATH = 'swear_predictor_model.h5'
TOKENIZER_PATH = 'swear_tokenizer.pickle'
# !!! --- IMPORTANT: Set this to your optimal threshold --- !!!
PREDICTION_THRESHOLD = 0.50 # Replace with your actual threshold value
# ---

# --- Model & Tokenizer Loading ---
MAX_SEQUENCE_LENGTH = 60 # Should match the value used during training

# Load the model
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    # You might want to run a dummy prediction to fully initialize the model if needed
    # model.predict(np.zeros((1, MAX_SEQUENCE_LENGTH)))
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load the tokenizer
if not os.path.exists(TOKENIZER_PATH):
    print(f"Error: Tokenizer file not found at {TOKENIZER_PATH}")
    exit()
try:
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

# --- Text Cleaning (Should match training) ---
# Define the *exact same* clean_text function used during training
def clean_text(text):
    """Basic cleaning: lowercase, remove speaker tags like NAME (...)"""
    text = text.lower()
    # Remove speaker tags like "MAC (something):" or just "(something)"
    # Adjust regex if your training cleaning was different
    text = re.sub(r'^[a-z\s]+(\(.*\))?:', '', text) # Speaker at start
    text = re.sub(r'\([a-z\s.,!?]+\)', '', text) # Inline parentheticals
    text = text.strip()
    return text

# --- Speech Recognition Setup ---
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Adjust microphone sensitivity if needed (helps ignore background noise)
# recognizer.energy_threshold = 4000 # Experiment with values
# recognizer.dynamic_energy_threshold = True # Adjusts threshold automatically

print("-" * 50)
print("Speech Recognition Configuration:")
print(f" - Sample Rate: {microphone.SAMPLE_RATE}")
print(f" - Sample Width: {microphone.SAMPLE_WIDTH}")
print(f" - Using Microphone: Default") # Or specify device_index if needed
print(f" - Recognizer Energy Threshold: {recognizer.energy_threshold}")
print(f" - Prediction Threshold: {PREDICTION_THRESHOLD:.2f}")
print("-" * 50)

# --- Real-time Prediction Loop ---

current_phrase = ""

print("Adjusting for ambient noise...")
with microphone as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)
print("Ready! Start speaking (press Ctrl+C to stop)...")

try:
    while True:
        print("\nListening...")
        with microphone as source:
            try:
                # Listen for the first phrase and extract it into audio data
                # Adjust timeout and phrase_time_limit as needed
                # timeout=None means wait indefinitely for a phrase
                # phrase_time_limit=None means record until silence
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("Got audio, processing...")
            except sr.WaitTimeoutError:
                print("No speech detected for a while.")
                # Optional: Reset context after long pause?
                # current_phrase = ""
                continue # Go back to listening

        try:
            # Recognize speech using Google Web Speech API
            # Choose other engines with: recognizer.recognize_sphinx(audio), recognize_whisper(audio), etc.
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")

            # Append recognized text to the current phrase
            current_phrase += (" " + text if current_phrase else text)
            print(f"Current context: '{current_phrase}'")

            # --- Prepare text for the model ---
            cleaned_context = clean_text(current_phrase)
            if not cleaned_context:
                print("(Cleaned context is empty, skipping prediction)")
                continue

            # Tokenize and pad
            text_seq = tokenizer.texts_to_sequences([cleaned_context])
            text_pad = pad_sequences(text_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='pre', truncating='pre') # Use 'pre' truncation

            # --- Predict ---
            # Check if padding resulted in an empty sequence (can happen with only OOV words)
            if np.array_equal(text_pad, np.zeros((1, MAX_SEQUENCE_LENGTH))):
                 print("(Sequence resulted in all zeros after padding/OOV, skipping prediction)")
                 # Optional: Reset context if it seems only OOV words were spoken
                 # current_phrase = ""
                 continue

            prediction_prob = model.predict(text_pad, verbose=0)[0][0]
            is_swear_likely = prediction_prob > PREDICTION_THRESHOLD

            # --- Display Result ---
            print(f"Prediction Probability: {prediction_prob:.4f}")
            if is_swear_likely:
                print(">>> Prediction: SWEAR WORD LIKELY NEXT! <<<")
            else:
                print(">>> Prediction: Not a swear word likely next.")

            # Optional: Simple context management (e.g., keep last N words)
            # words = current_phrase.split()
            # if len(words) > MAX_SEQUENCE_LENGTH * 2: # Keep roughly double the sequence length
            #    current_phrase = " ".join(words[-(MAX_SEQUENCE_LENGTH*2):])
            #    print(f"(Context shortened to: '{current_phrase}')")


        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            # Optionally, break the loop or add more robust error handling
            # break

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    print("Real-time prediction finished.")