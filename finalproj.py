

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




st.sidebar.write("Navigate here!!")
res=st.sidebar.radio("",["Home","Glaucoma","Drusen","Diabetic retinopathy","About"])


if res=="Home":
    st.write("""
        #   SOLUT('eye')ON
             """)
    image1="finalimage.jpg"
    st.image(image1,use_column_width=True)
    st.write("This is a web application very useful for doctors and patients during this pandemic situations.")
    st.write("This web application depicts three types of eye diseases namely:")
    st.text("Glaucoma")
    st.text("Drusen")
    st.text("Diabetic retinpathy")
    
    st.write("# Glaucoma")
    st.write("Glaucoma is a condition that damages your eye's optic nerve. It gets worse over time. It's often linked to a buildup of pressure inside your eye. Glaucoma tends to run in families. You usually don’t get it until later in life.The increased pressure in your eye, called intraocular pressure, can damage your optic nerve, which sends images to your brain. If the damage worsens, glaucoma can cause permanent vision loss or even total blindness within a few years.Most people with glaucoma have no early symptoms or pain. Visit your eye doctor regularly so they can diagnose and treat glaucoma before you have long-term vision loss.If you lose vision, it can’t be brought back. But lowering eye pressure can help you keep the sight you have. Most people with glaucoma who follow their treatment plan and have regular eye exams are able to keep their vision.")
    st.write("# Drusen")
    st.write("Drusen are small yellow deposits of fatty proteins (lipids) that accumulate under the retina.The retina is a thin layer of tissue that lines the back of the inside of the eye, near the optic nerve. The optic nerve connects the eye to the brain. The retina contains light-sensing cells that are essential for vision.Having a few hard drusen is normal as you age. Most adultsTrusted Source have at least one hard drusen. This type of drusen typically does not cause any problems and doesn’t require treatment.Soft drusen, on the other hand, are associated with another common eye condition called age-related macular degeneration (AMD). It’s called “age-related” macular degeneration because it’s more common in people older than 60.As soft drusen get larger, they can cause bleeding and scarring in the cells of the macula. Over time, AMD can result in central vision loss.")
    st.write("# Diabetic retinopathy")
    st.write("Diabetic retinopathy is a diabetes complication that affects eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina).At first, diabetic retinopathy might cause no symptoms or only mild vision problems. But it can lead to blindness.The condition can develop in anyone who has type 1 or type 2 diabetes. The longer you have diabetes and the less controlled your blood sugar is, the more likely you are to develop this eye complication.Over time, too much sugar in your blood can lead to the blockage of the tiny blood vessels that nourish the retina, cutting off its blood supply. As a result, the eye attempts to grow new blood vessels. But these new blood vessels don't develop properly and can leak easily.")
    
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
        
elif res=="About":
    st.write("# About the Application and developer")
    st.write("This is a web application developed using python.The tools used for developing the app are Spyder IDE,Google Colab Notebooks,streamlit package to develop the web application and share.streamlit to deploy the web application developed. The web applciation is easily accessible to anyone who has the eye scanned reports handy so that they can check the type of eye disease they are having and thus consult a doctor accordingly.")
    st.write("Initially data sets for different types of eye diseases images are collected from Kaggle. Glaucoma disease detection can be done using colour fundus images, Drusen disease detection can be done using OCT(optical coherent images), and diabetic retinopathy can be detected using color fundus images. After collection of data sets from Kaggle the data is imported to google drive for accessing via Google colab Notebooks.The required packages and image are imported in the Colab Notebook.The images are pre processed using ImageGenerator method. Later a convolution neural network model is built using different layers.The training images are fit into the model.Later the model accuracy is calculated. The accuracy and loss curves are also plotted. For more critical information Confusion Matrix is also built. The built model is saved as .h5 file and it is used for developing the web application. So in the project the model file is included along with the python file. The predictions are made using a defined function and the function is called at required display.The web application is developed using streamlit package. Finally the project is cloned into git repository to deploy using share.streamlit so that anyone in the world can access the application with single click on the link")
    st.write("Developed by:")
    st.write("The application was developed by M.Jahnavi as part of capstone project for the Fast track fall semester 2020-2021 under the guidance of Prof.Asish Kumar Dalai.")
    st.image("Jahnavi.png")