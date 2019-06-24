import cv2
import numpy as np
import os



def distance(p1,p2):
    return (np.sum((p2-p1)**2))**0.5




def knn(X,Y,test,k=5):
    m=X.shape[0]
    d=[]
    for i in range(m):
        dist = distance(X[i],test)
        d.append((dist,Y[i]))
    d = np.array(sorted(d))[:,1]
    d = d[:k]
    t = np.unique(d,return_counts=True)
    idx = np.argmax(t[1])
    pred = int(t[0][idx])
    return pred





cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("C:/Users/ASUS/Desktop/p/ML/Face_Recignition-using-Knn/haarcascade_frontalface_alt.xml")

dataset_path = "C:/Users/ASUS/Desktop/p/ML/ML/Dataset/"
labels = []
class_id = 0
names = {}
face_data = []

for fx in os.listdir(dataset_path):
	if fx.endswith(".npy"):
		names[class_id] = fx[:-4]
		print("Loading file ",fx)
		data_item = np.load(dataset_path + fx)
		face_data.append(data_item)


		#create labels
		target = class_id*np.ones((data_item.shape[0],))
		labels.append(target)
		class_id +=1

X = np.concatenate(face_data,axis=0)
Y = np.concatenate(labels,axis=0)

test = []

while True:
	ret,frame = cam.read()
	if ret == False:
		print("Something went Wrong")
		continue
	key_pressed = cv2.waitKey(1)&0xFF
	if key_pressed == ord('q'):
		break

	faces = face_cascade.detectMultiScale(frame,1.3,5)
	if(len(faces)==0):
		continue
	for face in faces:
		x,y,w,h = face


		face_section = frame[y-10:y+h+10,x-10:x+w+10]
		face_section = cv2.resize(face_section,(100,100))

		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),0)
	
	

	# print("X ",X.shape)
	# print("Y ",Y.shape)

	test = face_section.flatten()
	pred = knn(X,Y,test)
	
	pred = names[pred]

	cv2.putText(frame,pred, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

	cv2.imshow('VideoCapture',frame)

cam.release()
cv2.destroyAllWindows()










