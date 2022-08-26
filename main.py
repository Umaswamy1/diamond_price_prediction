import streamlit as st
import pickle
from PIL import Image
import sklearn
import numpy as np
lr = pickle.load(open('lr.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))
st.title("DIAMOND PRICE PREDICTION")
image = Image.open('diamond_image.jpg')
st.image(image)
carat = st.slider('CARAT',0.2,5.01,0.2)
cut = st.selectbox('CUT : Fair-0 , Good-1 , Very Good-2 , Premium-3 , Ideal-4',[0,1,2,3,4])
color = st.selectbox('COLOR : J(worst)-0 , I-1 , H-2 , G-3 , F-4 , E-5 , D(best)-6',[0,1,2,3,4,5,6])
clarity = st.selectbox('CLARITY : I1(worst)-0 , SI2-1 , SI1-2 , VS2-3 , VS1-4 , VVS2-5 , VVS1-6 , IF(best)-7',[0,1,2,3,4,5,6,7])
depth = st.slider('DEPTH(Height of the diamond)',43,79,43)
table = st.slider('TABLE',43,95,43)
x = st.slider('X(Length in mm)',10.74)
y = st.slider('Y(Width in mm)',58.9)
z = st.slider('Z(Depth in mm)',31.8)
if st.button('PREDICT PRICE'):
    query=np.array([carat,cut,color,clarity,depth,table,x,y,z])
    query=query.reshape(1,9)
    st.title(int(lr.predict(query)[0]))








