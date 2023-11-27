import cv2

import os
import time

def faceCrop(image,name):
    face_cascade = './face.xml'
    cascade = cv2.CascadeClassifier(face_cascade)

    img = cv2.imread(image)

    faces = cascade.detectMultiScale(img)
    counter = 0

    for f in faces:
        x, y, w, h = [ v for v in f ]
        # cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))
        # sub_face = img[int(y*1.5)-y :int((y+h)*1.5), int((x*1.5)-x):int((x+w)*1.5)]
        
        sub_face = img[y:y+h,x:x+w]

        cv2.imwrite(name+'_face_'+str(counter)+'.jpg', sub_face)
        counter += 1

    return

if (1):
    time_start = time.monotonic()

image_dir = './images/group/'
output_dir = './images/crop/'

for entry in os.listdir(image_dir):
    output_file = output_dir + 'cropped_' + entry.split('.')[0]
    faceCrop(image_dir+entry,output_file)  
    
print('Runtime: (', '{0:.2f}'.format(time.monotonic()-time_start), ' seconds)', sep='')
