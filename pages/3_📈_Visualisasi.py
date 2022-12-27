import warnings
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import tree
import streamlit as st
from web_functions import train_model

image = Image.open('tree.png')
st.image(image, caption='Decision Tree')

def app(df, x, y):

    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Visualisasi Prediksi Anemia")

    if st.checkbox("Plot Confusion Matrix"):
        model, score = train_model(x,y)
        plt.figure(figsize=(10,6))
        ConfusionMatrixDisplay(model, x, y, values_format='d')
        st.pyplot()

    if st.checkbox("Plot Decision Tree"):
        model, score = train_model(x,y)
        dot_data = tree.export_graphviz(
            decision_tree=model, max_depth=3, out_file=None, filled=True, rounded=True,
            feature_names=x.columns, class_names=['0', '1']
        )
        
        st.graphviz_chart(dot_data)


