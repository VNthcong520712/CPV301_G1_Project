import cv2
import time 
import face_recognition
import numpy as np
import pickle
from scipy.spatial import KDTree

def recognize_face_from_image(image, known_face_encodings, known_face_ids, tree=None):
	# Giảm kích thước ảnh để tăng tốc
	small_frame = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
	rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
	face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
	
	results = []
	for (top, right, bottom, left) in face_locations:
		# Chuyển đổi tọa độ về kích thước gốc
		top, right, bottom, left = [int(x * 2) for x in (top, right, bottom, left)]
		face_encoding = face_recognition.face_encodings(image, [(top, right, bottom, left)])[0]
		
		# Sử dụng KDTree để tìm kiếm nhanh
		dist, idx = tree.query(face_encoding, k=1)
		id_name = known_face_ids[idx] if dist < 0.4 else "Unknown"
		results.append(id_name)
	
	return results

if __name__ == "__main__":
	task = input("task:")
	if "train" in task:
		dataset_path = "D:\Documents\Learning\FPT\SU25\CPV\excersice\Project\Code\CPV\AI1901_face_dataset"
		print("Đang tải dữ liệu mẫu ...")
		known_face_encodings, known_face_ids = load_dataset(dataset_path)
		with open('face_features.pkl', 'wb') as f:
			pickle.dump({
				'encodings': known_face_encodings,
				'ids': known_face_ids
			}, f)
		print(f"Đã tải {len(known_face_encodings)} mẫu.")
	else:
		with open('face_features.pkl', 'rb') as f:
			data = pickle.load(f)
		known_face_encodings = data['encodings']
		known_face_ids = data['ids']
		tree = KDTree(known_face_encodings)  # Tạo KDTree một lần

		cam = cv2.VideoCapture(0)
		if not cam.isOpened():
			print("Error: Could not open video stream.")
			exit()
		
		classi = []
		frame_count = 0
		startdec = time.time()
		while True:
			start = time.time()
			
			rec, frame = cam.read()
			if not rec:
				print("Cannot read from camera. Exiting.")
				break
			
			frame_count += 1
			if frame_count % 3 != 0:  # Chỉ xử lý mỗi 3 khung hình
				continue
				
			ids = recognize_face_from_image(frame, known_face_encodings, known_face_ids, tree)
			if ids and "Unknown" not in ids:
				classi.extend(ids)
			if len(classi) > 5: 
				most = max(classi, key=classi.count)
				if classi.count(most) > 5:
					print("Detected:", most, time.time()-startdec)
					classi = []
					startdec = time.time()
			# print(1/(time.time() - start))
			
			cv2.imshow("Frame", frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		cam.release()
		cv2.destroyAllWindows()