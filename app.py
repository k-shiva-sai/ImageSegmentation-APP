import cv2
import face_recognition as f
import streamlit as st
import numpy as np
from PIL import Image
import mediapipe as mp
import os
import base64
from io import BytesIO

def get_image_download_link(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:png/jpg;base64,{img_str}">Download Image</a>'
    return href

mp_drawings=mp.solutions.drawing_utils
mp_face_mesh=mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

model_face_mesh=mp_face_mesh.FaceMesh()
st.title("STREAMLIT APP")
st.subheader("THIS APP PERFORMS OPERATIONS ON IMAGE LIKE SEGMENTATION, FACERECOGNITION")
selectbox = st.sidebar.selectbox(
    "MENU BAR",
    ("ABOUT","FACE RECOGNITION", "FACE DETECTION", "SELFIE SEGMENTATION")
)
if selectbox=="ABOUT":
    st.write("This app performs various rendering on images try by uploading an image")
elif selectbox=="FACE RECOGNITION":
    button=st.sidebar.radio("Detectable Faces",("Recognize or Run algorithm","Known Faces to Algorithm"))
    if button=="Recognize or Run algorithm":
        image_path=st.sidebar.file_uploader("Upload an Image")
        if image_path is not None:
            st.write("Uploaded Image will be cheeckd by the code written and recognizes the image")
            image=np.array(Image.open(image_path))
            st.sidebar.image(image)

            shiva=f.load_image_file("shiva.jpg")
            shiva_encode=f.face_encodings(shiva)[0]
            shiva_locate=f.face_locations(shiva)[0]

            mahesh=f.load_image_file("C:\\Users\\Shiva\\Desktop\\LETS UPGRADE\\pichai.jpg")
            mahesh_encode=f.face_encodings(mahesh)[0]
            mahesh_locations=f.face_locations(mahesh)[0]

            known_encode=[shiva_encode,mahesh_encode]
            known_faces=["shiva","pichai"]

            image_encode=f.face_encodings(image)
            image_location=f.face_locations(image)
            face_name=[]

            for encode in image_encode:
                match=f.compare_faces(known_encode,encode)
                name="Unkown"
                face_dst=f.face_distance(known_encode,encode)
                idx=np.argmin(face_dst)
                if match[idx]:
                    name=known_faces[idx]
                face_name.append(name)
            for (top,right,bottom,left),name in zip(image_location,face_name):
                cv2.rectangle(image,(left,top),(bottom,right),(114,20,245))
                font=cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(image,name,(left+10,bottom-10),font,0.75,(0,0,0),2)
            st.image(image)
            result=Image.fromarray(image)
            st.markdown(get_image_download_link(result), unsafe_allow_html=True)
        else:
            st.write("Image is not being uploaded. Please Upload an image to see the face Recognition")
    else:
        image1=cv2.imread("shiva_detected.jpg")
        image1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
        image2=cv2.imread("pichai_detected.jpg")
        image2=cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)
        st.image(image1)
        st.image(image2)
        st.write("These are the recognizable faces by the algorithm can also be exteneded to others")
    
elif selectbox=="FACE DETECTION":
    image_path=st.sidebar.file_uploader("Upload an Image")
    st.write("Detects faces on the images uploaded")
    if image_path is not None:
        image=np.array(Image.open(image_path))
        with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
            results=face_detection.process(image)
            annotated_image = image.copy()
            for detection in results.detections:
                #st.write('Nose tip:')
                #st.write(mp_face_detection.get_key_point(
                #    detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
                mp_drawings.draw_detection(annotated_image, detection)
            st.image(annotated_image)
            result=Image.fromarray(annotated_image)
            st.markdown(get_image_download_link(result), unsafe_allow_html=True)
            st.write("Face is being detected for the given image")
    else:
        st.write("Image is not being uploaded. Please Upload an image to see the Face Detection")
elif selectbox=="SELFIE SEGMENTATION":
    button=st.sidebar.radio("Color or Background",("None","RED","GREEN","IRON MAN BACKGROUND","AVENGERS BACKGROUND","OWN BACKGROUND IMAGE"))
    image_path=st.sidebar.file_uploader("Upload an Image")
    if button=="None":
        if image_path is not None:
            image=np.array(Image.open(image_path))
            st.sidebar.image(image)
            st.image(image)
            st.write("Choose any method to segment your image by chossing on respective button")
        else:
            st.write("Image is not being uploaded. Please Upload an image to see the segmentation")
    elif button=="RED":
        if image_path is not None:
            image=np.array(Image.open(image_path))
            st.sidebar.image(image)
            r,g,b=cv2.split(image)
            zeros=np.zeros(image.shape[:2],dtype="uint8")
            blended=cv2.merge([r,zeros,zeros])
            st.image(blended)
            result=Image.fromarray(blended)
            st.markdown(get_image_download_link(result), unsafe_allow_html=True)
            st.write("Given Image's background is changed to red")
        else:
            st.write("Image is not being uploaded. Please Upload an image to see the segmentation")
    elif button=="GREEN":
        if image_path is not None:
            image=np.array(Image.open(image_path))
            st.sidebar.image(image)
            r,g,b=cv2.split(image)
            zeros=np.zeros(image.shape[:2],dtype="uint8")
            blended=cv2.merge([zeros,g,zeros])
            st.image(blended)
            st.write("Given Image's background is changed to red")
            result=Image.fromarray(blended)
            st.markdown(get_image_download_link(result), unsafe_allow_html=True)
        else:
            st.write("Image is not being uploaded. Please Upload an image to see the segmentation")
    elif button=="IRON MAN BACKGROUND":
        if image_path is not None:
            st.write("Sliders are there in the side bar to choose the intensity of corresponding image")
            image=np.array(Image.open(image_path))
            ironman=cv2.imread("iron.jpg")
            ironman=cv2.resize(ironman,(image.shape[1],image.shape[0]))
            ironman=cv2.cvtColor(ironman,cv2.COLOR_BGR2RGB)
            image_intensity=st.sidebar.select_slider("MAIN IMAGE INTENSITY",options=[0.5,0.6,0.7,0.8,0.9,1])
            background=st.sidebar.select_slider("Background Image Intensity",options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            alpha=st.sidebar.select_slider("opcaity",options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            blended=cv2.addWeighted(image,image_intensity,ironman,background,alpha)
            st.image(blended)
            result=Image.fromarray(blended)
            st.markdown(get_image_download_link(result), unsafe_allow_html=True)
        else:
            st.write("Image is not being uploaded. Please Upload an image to see the segmentation")
    elif button=="AVENGERS BACKGROUND":
        if image_path is not None:
            st.write("Sliders are there in the side bar to choose the intensity of corresponding image")
            image=np.array(Image.open(image_path))
            ironman=cv2.imread("avengers.jpg")
            ironman=cv2.resize(ironman,(image.shape[1],image.shape[0]))
            ironman=cv2.cvtColor(ironman,cv2.COLOR_BGR2RGB)
            image_intensity=st.sidebar.select_slider("MAIN IMAGE INTENSITY",options=[0.5,0.6,0.7,0.8,0.9,1])
            background=st.sidebar.select_slider("Background Image Intensity",options=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            alpha=st.sidebar.select_slider("opcaity",options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            blended=cv2.addWeighted(image,image_intensity,ironman,background,alpha)
            st.image(blended)
            result=Image.fromarray(blended)
            st.markdown(get_image_download_link(result), unsafe_allow_html=True)
        else:
            st.write("Image is not being uploaded. Please Upload an image to see the segmentation")
    elif button=="OWN BACKGROUND IMAGE":
        back_img_path=st.sidebar.file_uploader("Please Upload Your own Background Image")
        if image_path is not None:
            st.write("Sliders are there in the side bar to choose the intensity of corresponding image")
            image=np.array(Image.open(image_path))
            if back_img_path is None:
                st.image(image)
                st.write("Please Upload your background Image")
            else:
                back_img=np.array(Image.open(back_img_path))
                back_img=cv2.resize(back_img,(image.shape[1],image.shape[0]))
                image_intensity=st.sidebar.select_slider("MAIN IMAGE INTENSITY",options=[0.5,0.6,0.7,0.8,0.9,1])
                background=st.sidebar.select_slider("Background Image Intensity",options=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
                alpha=st.sidebar.select_slider("opcaity",options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
                blended=cv2.addWeighted(image,image_intensity,back_img,background,alpha)
                st.image(blended)
                result=Image.fromarray(blended)
                st.markdown(get_image_download_link(result), unsafe_allow_html=True)
        else:
            st.write("Image is not being uploaded. Please Upload an image to see the segmentation")