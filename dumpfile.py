import cv2
import numpy 
fobject=cv2.CascadeClassifier("bello.xml")
recon=cv2.face.LBPHFaceRecognizer_create()
img=cv2.imread("benedict_cumberbatchjpg.jpg",0)
#img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=numpy.array(img,"uint8")
print(type(img))
#cv2.imshow("",img)
#cv2.waitKey(0)
totalroi=[]
label=[]
facesd=fobject.detectMultiScale(img,scaleFactor=1.3,minNeighbors=5)
for x,y,w,h in facesd:
        roi=img[y:y+h,x:x+w]
        totalroi.append(roi)
        label.append(0)

recon.train(totalroi,numpy.array(label))
recon.save("Train.yml")