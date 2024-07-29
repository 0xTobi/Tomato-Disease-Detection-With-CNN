import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_tomato_disease_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("TOMATO DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Tomato Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About the CNN Model
                A Convolutional Neural Network (CNN) is a deep learning algorithm designed for image recognition and classification tasks. CNNs are particularly effective in capturing spatial hierarchies in images through their convolutional layers, which apply various filters to the input data to detect patterns such as edges, textures, and shapes.
                
                In this project, the CNN model is trained to classify tomato leaves into 10 different categories, including both healthy and diseased states. By using a CNN, the model can accurately identify diseases from images, providing a valuable tool for early disease detection and management in agriculture.
                
                #### About Dataset
                This dataset was generated through offline augmentation from the original dataset, which is available on this GitHub repo. It comprises 26,503 RGB images of healthy and diseased tomato leaves, categorized into 10 classes. The dataset is split 80/20 into training and validation sets, with 21,043 images (80%) for training and 5,460 images (20%) for validation, maintaining the directory structure.

                The dataset combines two sources:
                https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf/data
                https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data
                
                A separate directory with 16 test images was created for prediction purposes.
                
                #### Content
                1. train (21,043 images)
                2. validation (5,460 images)
                3. test (16 images)
                """)
    
#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))