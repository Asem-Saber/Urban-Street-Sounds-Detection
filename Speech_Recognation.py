import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model and any necessary preprocessing functions or modules
model = load_model("Urban Street Sounds Classification.h5")

# Define a function to extract features from the audio file
def extract_features(data):
    # Zero Crossing Rate
    # the temporal properties of an audio signal 
    #The ZCR is a measure of the number of times the audio signal changes sign per unit time
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr))

    # Chroma_stft
    #It is a useful feature for characterizing the tonal properties
    #the chroma feature is a 12-element vector that represents the relative intensity of each of the 12 pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=22050, n_fft=200).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=22050, n_fft=200).T, axis=0)
    result = np.hstack((result, mfcc))

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=22050, n_fft=200).T, axis=0)
    result = np.hstack((result, mel))

    # Tonnetz
    tonnetz = np.mean(librosa.feature.tonnetz(y=data, sr=22050).T, axis=0);
    result = np.hstack((result, tonnetz))

    return result

# Create a streamlit app with a file uploader widget
st.title('Sound Classification App')
uploaded_file = st.file_uploader('Choose an audio file', type=['wav'])
classes=["air_conditioner","car_horn","children_playing","dog_bark","drilling","engine_idling","gun_shot","jackhammer","siren","street_music"]
if uploaded_file is not None:
    # Use the uploaded audio file to extract the necessary features
    audio_data, _ = librosa.load(uploaded_file, sr=44100, mono=True)
    features = extract_features(audio_data)
    print(features.shape)
    pred_fea = np.expand_dims(features, axis=0)  # Add batch dimension
    pred_fea = np.expand_dims(pred_fea, axis=2)  # Add input dimension
    pred_vector = np.argmax(model.predict(pred_fea), axis=-1)

    # Display the uploaded audio file
    st.audio(uploaded_file)

    # Display the prediction results to the user
    st.write('Predicted class:', classes[pred_vector[0]])