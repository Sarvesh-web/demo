import cv2
import pickle
vid=cv2.VideoCapture(0) #0 idicates the camera number
fobject=cv2.CascadeClassifier("bello.xml")
recon=cv2.face.LBPHFaceRecognizer_create()
recon.read("Trainner.yml")
lable={}
with open("labels.pickle",'rb') as f:
    og_lables=pickle.load(f)
    lable = {v:k for k,v in og_lables.items()}
while(True):
    c,f=vid.read()
    fgray=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    facesd=fobject.detectMultiScale(fgray,scaleFactor=1.2,minNeighbors=5)
    for x,y,w,h in facesd:
        cv2.rectangle(f,(x,y),(x+w,y+h),(0,255,0),2)
        roi=fgray[y:y+h,x:x+w]
        i,conf=recon.predict(roi)
        if conf>=45 :
            name=lable[i]
            cv2.putText(f,name,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        #if efficiency>60:
            #print("face matched")
        #v2.imshow("ROI",roi)
    cv2.imshow("FaceDetector",f)
    k=cv2.waitKey(20)
    if k==ord("q"):
        break
vid.release()
cv2.destroyAllWindows()