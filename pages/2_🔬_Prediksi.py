from sre_constants import SUCCESS
import streamlit as st
import pickle
import numpy as np

st.title("Input Data Prediksi Penyakit Anemia")

# Load Save Model
model = pickle.load(open('penyakit_anemia.sav', 'rb'))

homogoblin = st.text_input('Nilai Homogoblin')
mch        = st.text_input('Nilai MCH')
mchc       = st.text_input('Nilai MCHC')
mcv        = st.text_input('Nilai MCV')

# Kode untuk Prediksi
diagnosa_anemia = ''

# Membuat tombol Prediksi
if st.button('Prediksi Penyakit Kanker'):
    prediksi_anemia = model.predict([[homogoblin, mch, mchc, mcv ]])
    if (prediksi_anemia[0] == 1):
        diagnosa_anemia = 'Pasien mengidap Anemia'
    else:
        diagnosa_anemia = 'Pasien tidak mengidap Anemia'

st.success(diagnosa_anemia)
