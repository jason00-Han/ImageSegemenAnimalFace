from pyngrok import ngrok
ngrok.set_auth_token('2YFMu0fnHyjSdIjm6tTuLUtnbFn_6kkmXv7HD6SEY35qhNxVa')

%%writefile app.py
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.applications.imagenet_utils import decode_predictions

resnet50_pre = tf.keras.applications.ResNet50(weights='imagenet', input_shape=(224, 224, 3))
st.title('이미지 분류 인공지능 웹페이지')

file = st.file_uploader('이미지를 업로드하세요', type=['jpg', 'png'])
if file is  None:
    st.text('이미지를 먼저 올려주세요.')
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    img_resized = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
    img_resized = img_resized.convert('RGB')
    img_resized = np.array(img_resized)
  
    # resnet50_pre.predict()와 decode_predictions 함수 처리
  
    pred = resnet50_pre.predict(img_resized.reshape(1, 224, 224, 3))
    decoded_pred = decode_predictions(pred)
    results = ''
    for i, instance in enumerate(decoded_pred[0]):
        results += '{}위 {} ([2f]%) '.format(i+1, instance[1], instance[2]*100)
    st.success(results)

!streamlit run app.py &>/content/logs.txt & npx localtunnel --port 8501& curl ipv4.icanhazip.com
