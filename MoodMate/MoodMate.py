import openai
import random
import pandas as pd
import nltk
import pickle
import re
import librosa
import soundfile
import whisper
import os, glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from textblob import Word
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords 
from pydub import AudioSegment
from sklearn.preprocessing import LabelEncoder 
#"STOPWORDS" una lista di parole in inglese considerate stopwords.
STOPWORDS = stopwords.words("english")

# Load the serialized label encoder object from the file
with open('E:\Python\TelegramBotDepression\label_encoder.pickle', 'rb') as handle:
    Le = pickle.load(handle)


QUESTION = ["Have you felt hopeless or depressed for several weeks or months?",
            "Do you struggle to engage in activities you once found enjoyable?",
            "Are you feeling tired or low on energy?",
            "Do you have trouble maintaining a normal sleep schedule?",
            "Do you feel either restless or sluggish during the day?",
            "Have your eating habits changed?",
            "Do you struggle to concentrate at work, school, or daily activities?",
            "Are you worried that you let yourself or your loved ones down?",
            "Do these issues cause problems between you and your loved ones?",
            "Do you go out with your friends? Do you enjoy that time?"]

ANSWER = ["Remember, this depression test is meant only as a self-assessment. A professional evaluation can help you determine what services are right for you.",
          "One or two days, or even a week of feeling down or depressed may not lead to a diagnosis of depression.",
          "If you have depression, you may feel tired even if you have gotten plenty of sleep.",
          "People suffering from depression may not sleep enough or they may find themselves sleeping or napping more than usual",
          "Depression is not the same for everyone.",
          "Any major shift in eating habits, coupled with feeling down or depressed, can be a sign of clinical depression.",
          "Depression leaves you struggling to focus even on simple tasks.",
          "If you suffer from depression, you may worry that your friends and loved ones constantly feel let down.",
          "During long periods of depression, your loved ones may start to nitice the signs and symptoms and become worried or even angry."]

class Profile:
  depressed = 0
  not_depressed = 0

# Autenticazione
openai.api_key = "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"

# Carica il modello pre-addestrato
multinomial_model = pickle.load(open("E:\Python\TelegramBotDepression\model.pkl", "rb"))
# Load the CountVectorizer used to generate the feature vectors for the training data
vectorizer = pickle.load(open("E:\Python\TelegramBotDepression\idfvectorizer.pkl", "rb"))

sadness_counter = 0

def clean(text):
    text = text.lower()#converte testo in minuscolo
    text = re.sub("[^\w\s]","",text) # rimuove punteggiatura
    text = " ".join(w for w in text.split() if w not in STOPWORDS)# "join" unisce tutte le parole nella stringa che non sono presenti nella lista di stopwords.
    text = " ".join([Word(word).lemmatize() for word in text.split()])#utilizza la libreria NLTK per trasformare la parola in una forma base o radice, il che permette di trattare parole simili come una sola parola.
    return text

#funzione per rimuovere i caratteri duplicati consecutivamente all'interno della stessa parola
def remove_duplicates(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

#Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

def predictvoicetonesentiment(fileposition):
  filename = 'E:\Python\TelegramBotDepression\modelForPrediction1.sav'
  loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage

  # Load the M4A file
  #m4a_file = AudioSegment.from_file(fileposition, format="m4a")

  # Export the audio to WAV file
  #m4a_file.export("E:\Python\TelegramBotDepression\output.wav", format="wav")
  
  feature=extract_feature(fileposition, mfcc=True, chroma=True, mel=True)

  feature=feature.reshape(1,-1)

  prediction=loaded_model.predict(feature)
  prediction[0]

  if prediction[0] == "neutral" or prediction[0] == "happiness":
    model = whisper.load_model("base")
    result = model.transcribe(fileposition, fp16=False)

    text = result["text"]

    sentences = sent_tokenize(text)

    sadness_count = 0
    happiness_count = 0
    neutral_count = 0

    for sentence in sentences:
      if predictsentiment(sentence) == "sadness":
        sadness_count += 1
      elif predictsentiment(sentence) == "happiness":
        happiness_count += 1
      else:
        neutral_count += 1

    if sadness_count > happiness_count and sadness_count > neutral_count:
      trueprediction = "sadness"
    else:
      trueprediction = "neutral"
    return trueprediction
  else:
    return prediction[0]



def predictsentiment(text):
  # Pre-elaborare il testo in input
  inputdf = pd.DataFrame({'text': [text]})
  inputdf['text'] = inputdf['text'].apply(lambda x : clean(x))
  inputdf['text'] = inputdf['text'].apply(lambda x : remove_duplicates(x))
  tf = TfidfVectorizer(analyzer='word',max_features=10000,ngram_range=(1,3))
  text_vectors = vectorizer.transform(inputdf['text'])

  # Fare una previsione sul testo in input
  prediction = multinomial_model.predict(text_vectors)
  prediction = Le.inverse_transform([prediction[0]])
  proba =  np.max(multinomial_model.predict(text_vectors))

  # Stampa la classe prevista
  return prediction[0]


# Modello utilizzato
model_engine = "text-davinci-003"

def question(question):
  print(question)

  user_input = input("You: ")
  user_input.lower()

  if user_input == "yes":
    return True
  else:
    return False

def generate_response(prompt):
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = response.choices[0].text
    return message

class Profile:
  depressed = 0
  not_depressed = 0

def depression_test():
  p = Profile()
  answers = []

  for i in range(len(QUESTION)):
    if question(QUESTION[i]):
      answers.append(True)
    else:
      answers.append(False)
    
  for answer in answers:
    if answer:
      p.depressed += 1
    else:
      p.not_depressed += 1

  if p.depressed > p.not_depressed:
    print("\nMhhh, I think that you can be depressed...\n")
  else:
    print("\nIt's all right! You are not depressed!\n")

while True:
  print("Welcome to dIAry!"
  +"\nYou can text with me just asking something..."
  +"\n ...or you can take a quiz!"
  +"\n You can also talk to me, and I'll understand your feelings!"
  +"\n\n --- List of commands ---"
  +"\nquiz - start a 10 question quiz about your life"
  +"\nvoice - describe me your day in 30 seconds"
  +"\nexit - exit the chat\n\n")

 
 
  user_input = input("\nYou: ")

  sentiment = predictsentiment(user_input)

  if sentiment == "sadness":
    sadness_counter += 1

  if sadness_counter == 3:
    depression_test()
    print(f"\nDid you know that...\n{ANSWER[random.randint(0,8)]}\n")
  
  if sadness_counter > 5:
    print("\nMhhh, I'm sorry, i think that you can be sad...\n")

  if user_input == "exit":
    # esci dalla chat
    break
      
  elif user_input == "quiz":
    # avvia il quiz
    depression_test()

    print(f"\nDid you know that...\n{ANSWER[random.randint(0,8)]}\n")

  elif user_input == "voice":
    # avvia sentiment recognition
    print("Ok, I'm ready")
    sentimentvoice = predictvoicetonesentiment('Test1.wav')
    if sentimentvoice == "sadness":
      print("\nMhhh, I'm sorry, i think from your voice that you can be sad... try answer to this quiz\n")
      depression_test()
      print(f"\nDid you know that...\n{ANSWER[random.randint(0,8)]}\n")
    if sentimentvoice == "happiness" or sentimentvoice == "neutral":
      print("\nIt's all right! You are not sad!\n")




  else:
    # risposta del bot
    response = generate_response(user_input)
    print("Chatbot: ", response)