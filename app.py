import streamlit as st
from fastai.vision.all import *
import pathlib

import plotly.express as px

# import platform
# plt = platform.system()
# if plt=='Linux':pathlib.WindowsPath=pathlib.PosixPath

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.header('Classification')


select = st.selectbox('Klassifikasiya turini tanlang', ('None', 
                                  'Ayiqlar', 
                                  'GM_Avto'), on_change=None)

if select=='Ayiqlar' :
    
    st.info("Qutb_Ayigi, Qo'ng'ir_Ayiq, Panda va O'yinchoq_Ayiq turlarini klassifikasiya qilish")
    
    file = st.file_uploader("Rasmni kiriting",
                            type = ['png',
                                    'jpg',
                                    'jpeg',
                                    'svg'])

    if file:
        st.image(file)
        img = PILImage.create(file)
        
        classification_model = load_learner('bear2.pkl')
        
        prediction, pred_id, prob = classification_model.predict(img)
        
        st.success(f'Bashorat: {prediction}')
        st.info(f'Ehtimollik: {prob[pred_id]*100:.1f} %')
        
        fig = px.bar(x=prob*100,
                     y=classification_model.dls.vocab)
            
        st.plotly_chart(fig)

if select=='GM_Avto':
    st.info("Cobalt, Gentra, Damas va Matiz GM_AVTO turlarini klassifikasiya qilish")
    
    file = st.file_uploader("Rasmni kiriting",
                            type = ['png',
                                    'jpg',
                                    'jpeg',
                                    'svg'])

    if file:
        st.image(file)
        img = PILImage.create(file)
        
        classification_model = load_learner('avto.pkl')
        
        prediction, pred_id, prob = classification_model.predict(img)
        
        st.success(f'Bashorat: {prediction}')
        st.info(f'Ehtimollik: {prob[pred_id]*100:.1f} %')
        
        fig = px.bar(x=prob*100,
                     y=classification_model.dls.vocab)
            
        st.plotly_chart(fig)

      


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    