import cv2
import numpy as np
from sklearn.svm import SVC
harr_data = cv2.CascadeClassifier('C:\\Users\\HP\\Desktop\\datany\\cd\\haarcascade_frontalface_default.xml')

#captures realtime images from camera and collects the image data
#------
capture = cv2.VideoCapture(0)
data = []
while True:
    
    flag, img = capture.read()
    if flag:
        
        
        faces = harr_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,80,255),4)
            sqimg = img[y:y+h, x:x+w,:]
            sqimg = sqimg = cv2.resize(sqimg,(850,850))
            print(len(data))
            if len(data)<=100:
                data.append(sqimg)
            
            
  
        cv2.imshow('uo',img)
        if cv2.waitKey(2) == 27 or len(data)>101:
            break
 


capture.release()
cv2.destroyAllWindows()
##-------end 
np.save('data_', data) #data stored which is used to recognition
