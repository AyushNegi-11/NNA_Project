import streamlit as st
import numpy as np
import json
from PIL import Image
import io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

# CSS for aesthetic improvements
st.set_page_config(page_title="Image Captioning AI", layout="wide", page_icon="🖼️")

st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
    font-family: 'Inter', sans-serif;
}
.stApp header {
    background-color: transparent;
}
.title-text {
    font-size: 3rem;
    font-weight: 800;
    background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
    padding-bottom: 0px;
    text-align: center;
}
.subtitle-text {
    font-size: 1.2rem;
    color: #6c757d;
    text-align: center;
    margin-bottom: 30px;
}
.caption-box {
    padding: 20px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-top: 20px;
    text-align: center;
    font-size: 24px;
    font-weight: 500;
    color: #343a40;
    transition: transform 0.2s;
}
.caption-box:hover {
    transform: translateY(-5px);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title-text'>Image Captioning AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Upload an image and let AI describe it for you!</p>", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_tokenizer():
    # Load the InceptionV3 feature extractor exactly as in notebook
    model_incept = InceptionV3(weights='imagenet')
    feature_extractor = Model(model_incept.input, model_incept.layers[-2].output)
    
    # Load the tokenizer
    with open("tokenizer.json", "r") as f:
        tokenizer_data = json.load(f)
    
    vocab_size = int(tokenizer_data['vocab_size'])
    max_length = int(tokenizer_data['max_length'])
    embedding_dim = 200
    
    # Build caption model architecture to avoid custom layer loading issues
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # Load weights from the original full model H5 file to ensure exactly correct epoch weights are used!
    caption_model.load_weights("image_captioning_model.h5")
    
    return feature_extractor, caption_model, tokenizer_data

try:
    with st.spinner("Loading AI Models... Please wait."):
        feature_extractor, caption_model, tokenizer_data = load_models_and_tokenizer()
        wordtoix = tokenizer_data['wordtoix']
        ixtoword = tokenizer_data['ixtoword']
        max_length = int(tokenizer_data['max_length'])
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

def preprocess_image(uploaded_file):
    # Match the notebook prepocess perfectly by saving to file then load_img
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    img = load_img("temp.jpg", target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode_image(uploaded_file):
    processed_img = preprocess_image(uploaded_file)
    fea_vec = feature_extractor.predict(processed_img, verbose=0)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

def greedy_search(photo_features):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([np.array([photo_features]), sequence], verbose=0)
        yhat = np.argmax(yhat)
        
        word = ixtoword.get(str(yhat))
        if word is None:
            break
            
        in_text += ' ' + word
        if word == 'endseq':
            break
            
    final = in_text.split()
    if len(final) > 2:
        final = final[1:-1]
    else:
        final = final[1:]
    return ' '.join(final).capitalize()

def beam_search_predictions(photo_features, beam_index=3):
    start = [wordtoix["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_length, padding='post')
            preds = caption_model.predict([np.array([photo_features]), par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                
        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word = start_word[-beam_index:]
        
    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword.get(str(i), '') for i in start_word]
    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break
            
    # filter out empty strings and startseq
    final_caption = [w for w in final_caption if w and w != 'startseq']
    return ' '.join(final_caption).capitalize()

col1, col2 = st.columns([1, 1])

with st.sidebar:
    st.markdown("### Settings")
    decoding_method = st.selectbox(
        "Decoding Method",
        ["Greedy Search", "Beam Search (k=3)", "Beam Search (k=5)"]
    )

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Pass the bytes buffer directly to ensure identical preprocessing logic
        photo_features = encode_image(uploaded_file)
        
        # Open with PIL just for display
        display_image = Image.open(uploaded_file)
        
        with col1:
            st.image(display_image, caption='Uploaded Image', use_container_width=True)
            
        with col2:
            st.markdown("### Generating Caption...")
            with st.spinner("AI is thinking..."):
                if decoding_method == "Greedy Search":
                    caption = greedy_search(photo_features)
                elif decoding_method == "Beam Search (k=3)":
                    caption = beam_search_predictions(photo_features, beam_index=3)
                else:
                    caption = beam_search_predictions(photo_features, beam_index=5)
                
            st.markdown(f"<div class='caption-box'>✨ {caption}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
