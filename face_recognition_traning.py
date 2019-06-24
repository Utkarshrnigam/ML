import cv2
import numpy as np

# read a video stream and display it.

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("C:/Users/ASUS/Desktop/p/ML/Face_Recignition-using-Knn/haarcascade_frontalface_alt.xml")

face_data = []
c = 0
k=0
name = input("enter user name: ")
while True:
	c = c+1
	ret,frame = cam.read()
	key_pressed = cv2.waitKey(1)&0xFF
	if ret==False:
		print("Something went wrong")
		continue
	if key_pressed == ord('q'):
		break

	
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	# bright_img = frame + 50
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	if(len(faces)==0):
		continue
	for face in faces:
		x,y,w,h = face
		# cv2.rectangle(gray,(x,y),(x+w,y+h),(0,255,255),0)

		face_section = frame[y-10:y+h+10,x-10:x+w+10]
		face_section = cv2.resize(face_section,(100,100))
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),0)

	if c%10 == 0:
		face_data.append(face_section)
		print("taking image",c)
		k = k+1

	if k == 17:
		break


	# cv2.imshow('VideoCapture_bright',bright_img[:,::-1,:])
	cv2.imshow('VideoCapture',frame[:,::-1,:])
	cv2.imshow('face_secton',face_section[:,::-1,:])
	# cv2.imshow('VideoCapture_gray',gray[:,::-1])  #cannot merge gray and image as gray image ha only on channel

	# new_3channel_gray = np.zeros((*gray.shape,3))
	# new_3channel_gray[:,:,0] = gray
	# new_3channel_gray[:,:,1] = gray
	# new_3channel_gray[:,:,2] = gray

	# combined_image = np.hstack((frame,new_3channel_gray))
	# cv2.imshow('VideoCapture',combined_image[:,::-1,:])

print('Total_faces: ',len(face_data))
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0],-1)) 


np.save('./Dataset/'+name+'.npy',face_data)
print(face_data.shape)
	


cam.release()
cv2.destroyAllWindows()