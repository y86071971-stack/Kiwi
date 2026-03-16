import streamlit as st 
import tensorflow as tf
from PIL import Image
import numpy as np

#load model
model = tf.keras.models.load_model("C:/Users/PC1/Desktop/kiwi/kiwi.h5")
#load image
upload_image = st.file_uploader("select image" , type=['jpg' , 'jpeg' ,'png'])
class_names = ["Kiwi B","Kiwi C","Kiwi A"]
#             [10 , 7 , 3 , 10 , 55 , 10 ,5 ]
if upload_image is not None :
        image= Image.open(upload_image)
        st.image(image , caption = "image uploader" )
# preprocessing image
        if image.mode != "RGB":
           image = upload_image.convert("RGB")

        img = image.resize((64,64))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img , axis=0)

# pred
        pred = model.predict(img)
        index = np.argmax(pred)

        st.info(class_names[index])