

import numpy as np
import streamlit as st
import tensorflow as tf
import os
from PIL import Image, ImageOps


from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



# Model saved with Keras model.save()
MODEL_PATH ='glmodel.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def import_and_predict(img_path, model):
    
       
    size = (224,224) 
    img = ImageOps.fit(img_path,size)
        
    image = img.convert('RGB')
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)

    img_reshape = image[np.newaxis,...]

    prediction = model.predict(img_reshape)
        
    return prediction





res=st.sidebar.radio("Navigate here!",["Home","Glaucoma","Drusen","Diabetic retinopathy","About"])


if res=="Home":
    st.write("""
             # SOLUT('eye')ON
             """)
    st.write("displaying home")    
    
elif res=="Glaucoma":
    st.write("""
                 #   Glaucoma detection
             """
                 )

  
    file = st.file_uploader("Please upload an image file", type=["jpg", "png","jpeg"])
        #
    if file is None:
            st.text("You haven't uploaded an image file")
        
    else:
            image = Image.open(file)
            st.image(image, use_column_width=True)
            
            prediction = import_and_predict(image, model)
           
            if prediction[0][1]>0.5:
                st.write("It is a normal image!Don't worry")
            elif prediction[0][1]<0.5:
                st.write("Your eye has gluacoma, consult doctor immediately")
            else:
                st.write("give correct image!")
    
            st.text("Probability (0: glaucoma, 1: normal)")
            st.write(prediction)
        
elif res=="Drusen":
     st.write("""
                 #   Drusen detection
             """
                 )

     file1 = st.file_uploader("Please upload an image file", type=["jpg", "png","jpeg"])
        #
     if file1 is None:
            st.text("You haven't uploaded an image file")
        
     else:
            image = Image.open(file1)
            st.image(image, use_column_width=True)
            model=load_model("dmodelcnn.h5")
            prediction = import_and_predict(image, model)
           
            if prediction[0][1]>0.5:
                st.write("It is a normal image!Don't worry")
            elif prediction[0][1]<0.5:
                st.write("Your eye has drusen, consult doctor immediately")
            else:
                st.write("give correct image!")
    
            st.text("Probability (0: drusen, 1: normal)")
            st.write(prediction)

elif res=="Diabetic retinopathy":
    st.write("""
             #  Diabetic retinopathy detection
             """)
    file2=st.file_uploader("Please upload an image file",type=["jpg","png","jpeg"])
    if file2 is None:
        st.text("You haven't uploaded an image file")
    else:
        image=Image.open(file2)
        st.image(image,use_column_width=True)
        model=load_model("diabeticretinomodel.h5")
        prediction=import_and_predict(image,model)
        
        if prediction[0][1]>=0.5:
            st.write("Your eye  has symptoms of diabetic retinopathy")
        elif prediction[0][1]<0.5:
            st.write("Your eye doesn't symptoms of diabetic retinopathy") 
            
        st.text("Probability (0:No symptoms, 1: Symptoms)")
        st.write(prediction)
    
    