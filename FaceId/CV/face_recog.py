import os, sys
import json
import cv2
import numpy as np

from face_detection import face_extraction

HOME = os.path.dirname(__file__)
sys.path.append(HOME)
print("Root directory:", HOME)

F_PATH = os.path.abspath(os.path.join(HOME, "..\.."))
config_path = os.path.join(F_PATH,"config.json")
with open(config_path,'r') as js: 
    config = json.load(js)
cfg = config["FACE_RECOGNITION"]

CAM = config["CAM_ID"]
LBPH_MODEL_PATH = cfg["YML_FILE"]
LABEL_MAP_PATH = cfg["LABEL_FILE"]

IMG_SIZE = (100, 100)
CONFIDENCE_THRESHOLD = cfg["CONFIDENCE"]

def load_label_map(label_map_path):
    id2name = {}
    with open(label_map_path, "r", encoding="utf-8") as f:
        for line in f:
            idx, name = line.strip().split(",", 1)
            id2name[int(idx)] = name
    return id2name

def recognize_face(face_img, recognizer, id2name):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
    gray = cv2.resize(gray, IMG_SIZE)
    label_id, confidence = recognizer.predict(gray)
    if confidence < CONFIDENCE_THRESHOLD:
        return id2name.get(label_id, "Unknown"), confidence
    else:
        return "Unknown", confidence

if __name__ == "__main__":
	cam = cv2.VideoCapture(CAM)
	recog = cv2.face.LBPHFaceRecognizer_create()
	recog.read("D:\Documents\Learning\FPT\SU25\CPV\excersice\Project\Code\CPV\FaceId\CV\model_face_03-56-09.yml")
	id2name = load_label_map("D:\Documents\Learning\FPT\SU25\CPV\excersice\Project\Code\CPV\FaceId\CV\label_map.txt")
	try: 
		while True: 
			rec, frame = cam.read()
			if rec: 
				out = face_extraction(frame,True,padd=10)
				if len(out):
					cv2.imshow("face", out[0])
					name, conf = recognize_face(out[0], recog, id2name)
					print(f"Kết quả nhận dạng: {name} (Độ tin cậy: {conf:.2f})")
				else: 
					print(f"Kết quả nhận dạng: UNKNOWN")
				cv2.imshow("frame", frame)
			if (cv2.waitKey(1) & 0xFF) == ord('q'):
				break
	except KeyboardInterrupt:
		print("Existing")
	finally:
		print("Stop")
		cv2.destroyAllWindows()
		cam.release()
