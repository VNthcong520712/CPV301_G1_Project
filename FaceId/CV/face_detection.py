import os, sys
import json
import cv2
import numpy as np
import dlib 

HOME = os.path.dirname(__file__)
sys.path.append(HOME)
print("Root directory:", HOME)

F_PATH = os.path.abspath(os.path.join(HOME, "..\.."))
config_path = os.path.join(F_PATH,"config.json")
with open(config_path,'r') as js: 
    config = json.load(js)
cfg = config["FACE_DETECT"]

CAM = config["CAM_ID"]
PADD = cfg['PADD']
SHOW = cfg['SHOW_IMG']

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(F_PATH, "shape_predictor_68_face_landmarks.dat"))

def face_extraction(image, show=SHOW, padd=PADD): 
	img = np.copy(image)
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = detector(gray_image)
	out = []
	for face in faces:		
		# Detect landmarks
		landmarks = predictor(gray_image, face)
		shape = []
		for i in range(68):
			x = landmarks.part(i).x
			y = landmarks.part(i).y
			shape.append(dlib.point(x, y))
		
		# Draw result on frame
		x, y, w, h = face.left(), face.top(), face.width(), face.height()
	
		x1 = max(0, x - padd)
		y1 = max(0, y - padd)
		x2 = min(img.shape[1], x + w + padd)
		y2 = min(img.shape[0], y + h + padd)
		
		if show:
			img = cv2.rectangle(img, (x1,y1), (x2,y2), (200,200,10), 2)

		out.append(image[y1:y2, x1:x2,])
	
	return out
	

if __name__ == "__main__":
	cam = cv2.VideoCapture(CAM)
	try: 
		while True: 
			rec, frame = cam.read()
			if rec: 
				out = face_extraction(frame,True,padd=10)
				if len(out):
					cv2.imshow("frame",out[0])
				else: 
					cv2.imshow("frame", frame)
			if (cv2.waitKey(1) & 0xFF) == ord('q'):
				break
	except KeyboardInterrupt:
		print("Existing")
	finally:
		print("Stop")
		cv2.destroyAllWindows
		cam.release