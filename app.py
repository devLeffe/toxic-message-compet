

import streamlit as st
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd

@st.cache(allow_output_mutation=True)
def load_model():
    print('chargement model')
    model_eval = tf.keras.models.load_model(
        "models/BBB.h5", compile=True)
    # model_auto = load_model(‘models/auto_model.h5’, compile=False)
    with open('./models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    return model_eval, tokenizer

@st.cache(allow_output_mutation=True)
def init():
    sam=""

def generate_sample():
    df_sample=pd.read_csv('./models/sample.csv')
    pd.set_option('display.max_colwidth', None)

    sm=df_sample.sample(n=1)['comment_text'].to_string(index=False)
    print(sm)
    return sm

def predict():
    print(txt)
    model, tokenizer = load_model()

    text_preprocessing = tokenizer.texts_to_sequences([txt])
    text_preprocessing_pad = pad_sequences(text_preprocessing, maxlen=150)
    # text_preprocessing_pad
    print(model.predict(text_preprocessing_pad))
    return model.predict(text_preprocessing_pad)[0]


col1, col2 = st.columns([3, 1])
col1_1,col1_2,col1_3,col1_4,col1_5,col1_6=st.columns([1,1,1,1,1, 1])

sam=''
with col1:
    text_area = st.empty()

    txt = text_area.text_area('Text to analyze',sam,  placeholder='''
            Enter a text ...
            ''')

    button = st.button('Read 👓')
    metric = st.empty()
    
with col1_1:
    toxic_slider = st.empty()
    toxic_info = st.empty()
    
with col1_2:
    severe_toxic_slider = st.empty()
    severe_toxic_info = st.empty()
    
with col1_3:
    obsene_slider = st.empty()
    obsene_info = st.empty()

with col1_4:
    threat_slider = st.empty()
    threat_info = st.empty()

with col1_5:
    insult_slider = st.empty()
    insult_info = st.empty()

with col1_6:
    identity_hate_slider = st.empty()
    identity_hate_info = st.empty()
    

with col2:
    notebook = st.write("🙋🏽‍♂️ Author [Leffe Pierre](https://www.linkedin.com/in/pierre-leffe/)")
    notebook = st.write("📚 Notebook [Article](https://deepnote.com/@pierre-leffe-610b/Toxic-ccd6ffb7-14f3-4305-a281-ffe330e00717)")
    generate_button = st.button('♺  Generate')
    

if generate_button:
    samp=generate_sample()
    print(sam)
    print(type(sam))
    sam=samp
    txt=text_area.text_area('Text to analyze', f'''{sam}''', placeholder='''
        Enter a text ...
        ''')
    

if button:
    toxic,severe_toxic,obsene,threat,insult,identity_hate = predict()
    toxic=float(round(toxic,2)*100)
    severe_toxic=float(round(severe_toxic,2)*100)
    obsene=float(round(obsene,2)*100)
    threat=float(round(threat,2)*100)
    insult=float(round(insult,2)*100)
    identity_hate=float(round(identity_hate,2)*100)
    print('e=>',toxic)
    print('e=>',type(toxic))


    toxic_slider.slider(
        'Toxic',
        value=toxic,
        max_value=100.,
        min_value=0.,
        disabled=True,
        key = "toxic"
    )
    severe_toxic_slider.slider(
        'Severe toxic ',
        value=severe_toxic,
        max_value=100.,
        min_value=0.,
        disabled=True,
        key='severe_toxic'
    )
    obsene_slider.slider(
        'Obsene',
        value=obsene,
        max_value=100.,
        min_value=0.,
        disabled=True,
        key="obsene"
    )
    
    threat_slider.slider(
        'Threat',
        value=threat,
        max_value=100.,
        min_value=0.,
        disabled=True,
        key="threat"
    )
    insult_slider.slider(
        'Insult',
        value=insult,
        max_value=100.,
        min_value=0.,
        disabled=True,
        key="insult"
    )
    
    identity_hate_slider.slider(
        'Identity hate',
        value=identity_hate,
        max_value=100.,
        min_value=0.,
        disabled=True
    )

    if(toxic <40):
        toxic_info.success('✅  Correct')
    elif(toxic >=40 and toxic <60):
        toxic_info.info('ℹ️ just')
    elif(toxic >=60 and toxic <= 100):
        toxic_info.warning('❌ Anormal')
        
    if(severe_toxic <40):
        severe_toxic_info.success('✅  Correct')
    elif(severe_toxic >=40 and severe_toxic <60):
        severe_toxic_info.info('ℹ️ just')
    elif(severe_toxic >=60 and severe_toxic <= 100):
        severe_toxic_info.warning('❌ Anormal')
    
    if( obsene <40):
        obsene_info.success('✅  Correct')
    elif( obsene >=40 and  obsene <60):
        obsene_info.info('ℹ️ just')
    elif(obsene >=60 and obsene <= 100):
        obsene_info.warning('❌ Anormal')
    
    if( threat <40):
        threat_info.success('✅  Correct')
    elif( threat >=40 and  threat <60):
        threat_info.info('ℹ️ just')
    elif( threat >=60 and  threat <= 100):
        threat_info.warning('❌ Anormal')
    
    if(insult <40):
        insult_info.success('✅  Correct')
    elif(insult >=40 and insult <60):
        insult_info.info('ℹ️ just')
    elif(insult >=60 and insult <= 100):
        insult_info.warning('❌ Anormal')
        
        
    if( identity_hate <40):
        identity_hate_info.success('✅  Correct')
    elif( identity_hate >=40 and  identity_hate <60):
        identity_hate_info.info('ℹ️ just')
    elif( identity_hate >=60 and identity_hate <= 100):
        identity_hate_info.warning('❌ Anormal')
