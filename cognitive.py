import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
import base64

def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_img = get_base64("enhanced_brain_background.jpg")

page_bg_img = f'''
<style>
.stApp {{
background-image: url("data:image/jpg;base64,{bg_img}");
background-size: auto;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
image-rendering: auto;
background-colour: black;
}}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)


with open("cognitive.pkl","rb") as obj1:
    dict1=pickle.load(obj1)
    
st.title("🧠Cognitive Score Predictor")
st.write("Come , Let us check your cognitive score")

st.write("Check your Memory Test Score:")
st.write("https://www.totalbrain.com/mental-health-assessment/memory-test/")

st.write("Check your Reaction Time:")
st.write("https://humanbenchmark.com/tests/reactiontime")


st.write("Please fill in all the fields below before submitting.")
age=st.number_input("Age",15,65)

gender=st.selectbox("Gender",["Female","Male","Other"])
gender=dict1["label2"].transform([[gender]])[0]


sleep_dur=st.selectbox("Sleep Duration",["Reccomended Sleep","Long Sleep","Sleep Depreviation"])
sleep_dur=dict1["ord2"].transform([[sleep_dur]])
sleep_dur=sleep_dur.flatten()

stress=st.number_input("Stress Level",0,15)

diet=st.selectbox("Diet Type",["Non-Vegetarian","Vegetarian","Vegan"])
diet=dict1["onehot"].transform([[diet]])
diet=diet.flatten()

screentym=st.number_input("Daily screen Time in Hours",value=0.00,step=0.01,format="%.2f")

exercise=st.selectbox("Exercise Frequency",["Low","Medium","High"])
exercise=dict1["ord1"].transform([[exercise]])
exercise=exercise.flatten()

caffine=st.selectbox("Caffeine Intake (<400mg/day-Moderate)",["Moderate","High"])
caffine=dict1["label1"].transform([caffine])[0]

recationtym=st.number_input("Reaction Time in milliseconds",value=100.00,step=0.01,format="%.2f")

memory=st.number_input("Memory Test Score",20,100)

button=st.button("Predict")
if button:
    data=[[age,gender,*sleep_dur,stress,screentym,*exercise,caffine,recationtym,memory,*diet]]
    scaled=dict1["scaler"].transform(data)
    res=dict1["model"].predict(scaled)[0]
    st.success(f"Your Cognitive score is : {res}")
    if res<40:
        st.warning("Your score is below average level, you must focus more on your cognitive health, reduce your stress level and work on improving your concentration ")
