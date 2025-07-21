import os, sys
import json
import cv2
import numpy as np
import datetime

HOME = os.path.dirname(__file__)
sys.path.append(HOME)
print("Root directory:", HOME)

F_PATH = os.path.abspath(os.path.join(HOME, "..\.."))
config_path = os.path.join(F_PATH,"config.json")
with open(config_path,'r') as js: 
	config = json.load(js)
cfg = config["FEATURES_EXTRACT"]

CAM = config["CAM_ID"]
DATASET_DIR = cfg["DATA_PATH"]
OUTPUT_YML = cfg["OUT_YML"]

IMG_SIZE = (100,100)

def load_images_labels(dataset_dir):
	images = []
	labels = []
	label_map = {}
	current_id = 0 
	
	main_dir = [dir for dir in os.listdir(dataset_dir) if "." not in dir]
	for face_id in main_dir:
		id_direct = os.path.join(dataset_dir, face_id)
		for direct in os.listdir(id_direct):
			direct_dir = os.path.join(id_direct, direct)

			if face_id not in label_map: 
				label_map[face_id] = current_id
				current_id +=1
			
			label = label_map[face_id]
			for img_name in os.listdir(direct_dir):
				img_path = os.path.join(direct_dir, img_name)
				img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)	
				if img is None:
					continue
				img = cv2.resize(img, IMG_SIZE)
				images.append(img)
				labels.append(label)
	return images, np.array(labels), label_map

def features_extraction(dataset_dir=DATASET_DIR,output=OUTPUT_YML):
	images, labels, label_map = load_images_labels(dataset_dir)
	print(f"Loaded {len(images)} images, {len(label_map)} persons.")

	recog = cv2.face.LBPHFaceRecognizer_create()
	print("Training LBPH model...")
	recog.train(images, labels)
	current_time = datetime.datetime.now()
	name = f"model_face_{current_time.strftime('%H-%M-%S')}.yml"
	yaml = os.path.join(output, name)
	print(f"Saving features and labels to {output} ...")
	recog.save(yaml)
	with open(os.path.join(output,"label_map.txt"), "w", encoding="utf-8") as f:
		for name, idx in label_map.items():
			f.write(f"{idx},{name}\n")
	print("Done.")

if __name__ == "__main__":
	features_extraction(dataset_dir="D:\Documents\Learning\FPT\SU25\CPV\excersice\Project\Code\CPV\AI1901_face_dataset",
					 output="D:\Documents\Learning\FPT\SU25\CPV\excersice\Project\Code\CPV\FaceId\CV")