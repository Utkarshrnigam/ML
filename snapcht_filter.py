import cv2
import numpy as np

cam = cv2.VideoCapture(0)

eye_cascade = cv2.CascadeClassifier("C:/Users/ASUS/Desktop/p/ML/ML/Dataset/Got snapchat filter/third-party/frontalEyes35x16.xml")
mustach_cascade = cv2.CascadeClassifier("C:/Users/ASUS/Desktop/p/ML/ML/Dataset/Got snapchat filter/third-party/Nose18x15.xml")

glasses = cv2.imread("C:/Users/ASUS/Desktop/p/ML/ML/Dataset/Got snapchat filter/glasses.png",cv2.IMREAD_UNCHANGED)
moochh = cv2.imread("C:/Users/ASUS/Desktop/p/ML/ML/Dataset/Got snapchat filter/mustache.png",cv2.IMREAD_UNCHANGED)


while True:
	ret,frame = cam.read()
	key_pressed = cv2.waitKey(1)&0xFF
	if ret == False:
		print("Something went wrong")
		continue
	if key_pressed == ord('q'):
		break

	frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	eyes = eye_cascade.detectMultiScale(frame,1.3,5)
	noses = mustach_cascade.detectMultiScale(frame,1.3,5)
	if(len(eyes) == 0):
		continue
	else:
		for eye in eyes:
			x1,y1,w1,h1 = eye
			w1 += 10
			h1 += 10
			glasses = cv2.resize(glasses,(h1,w1))
			y2 = y1 + glasses.shape[0]
			x2 = x1 + glasses.shape[1]
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			alpha_s = glasses[:,:,3] / 255.0
			alpha_l = 1 - alpha_s 

			for c in range(0, 3):
				frame[y1:y2, x1:x2, c] = alpha_s * glasses[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c]

			cv2.rectangle(frame,(x1,y1),(x1+h1,y1+w1),(0,255,255),0)



		for nose in noses:
			x1,y1,w1,h1 = nose
			x1 = x1-40
			y1 = y1+25
			w1 = w1+100
			h1 = h1
			moochh = cv2.resize(moochh,(w1,h1))
			y2 = y1 + moochh.shape[0]
			x2 = x1 + moochh.shape[1]
			alpha_s = moochh[:,:,3] / 255.0
			alpha_l = 1.0 - alpha_s   
			for c in range(0, 3):
				frame[y1:y2, x1:x2, c] =  alpha_s*moochh[:, :, c] +alpha_l*frame[y1:y2, x1:x2, c]
			cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(0,255,255),0)


	cv2.imshow("eyes and nose",frame[:,::-1,:])

cam.release()
cv2.destroyAllWindows()
