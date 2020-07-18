import os
import cv2
import numpy as np 
from PIL import Image
import pickle
base_dir=os.path.dirname(os.path.abspath(__file__))
print(base_dir)
image_dir=os.path.join(base_dir, "Images")
recon=cv2.face.LBPHFaceRecognizer_create()
fobject=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
c_id =0
label_ids={}
y_labels=[]
x_train=[]

for root,dirs,files in os.walk(image_dir):
    for filex in files:
        if filex.endswith(".png") or filex.endswith("jpg"):
            path=os.path.join(root,filex)
            label= os.path.basename(root).lower()
            print(label,path)
            if not label in label_ids:
                label_ids[label]=c_id
                c_id+=1
            id_= label_ids[label]
            pil_image=Image.open(path).convert("L")
            image_array=np.array(pil_image,"uint8")
            facesd=fobject.detectMultiScale(image_array,scaleFactor=1.2,minNeighbors=5)
            for x,y,w,h in facesd:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)
recon.train(x_train,np.array(y_labels))
recon.save("Trainner.yml")