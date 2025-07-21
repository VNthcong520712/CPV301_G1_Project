import face_recognition
import pickle
import os
import sys
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Phần 1: Chuẩn bị dữ liệu (Training)
def train_faces(dataset_dir, output_file='student_encodings.pkl'):
	"""
	Huấn luyện mô hình nhận dạng khuôn mặt bằng cách trích xuất các encoding từ ảnh.

	Args:
		dataset_dir (str): Đường dẫn đến thư mục chứa dữ liệu ảnh của sinh viên.
						   Mỗi thư mục con trong dataset_dir là tên của một sinh viên,
						   và chứa các ảnh khuôn mặt của sinh viên đó.
		output_file (str): Tên file để lưu trữ các encoding đã trích xuất.
	"""
	known_encodings = {}  # key: tên sinh viên, value: danh sách các encoding
	n = 0
	# Duyệt qua từng thư mục của sinh viên
	for student_name in os.listdir(dataset_dir):
		# Bỏ qua các file không phải thư mục
		if "." in student_name:
			continue
		student_dir = os.path.join(dataset_dir, student_name)
		if os.path.isdir(student_dir):
			n += 1
			student_encodings = []
			print(f"{n}. Đang xử lý sinh viên: {student_name}")
			count = 0
			# Duyệt qua các thư mục con (nếu có) hoặc trực tiếp các ảnh
			for root, _, files in os.walk(student_dir):
				for img_file in files:
					# Chỉ xử lý các file ảnh phổ biến
					if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
						count += 1
						sys.stdout.write(f"\r  Đã xử lý {count} ảnh ({img_file})")
						sys.stdout.flush()
						# time.sleep(0.01)  # Giả lập xử lý mất thời gian, có thể bỏ nếu cần tốc độ
						img_path = os.path.join(root, img_file)
						try:
							# Tải ảnh
							image = face_recognition.load_image_file(img_path)
							# Phát hiện vị trí khuôn mặt
							face_locations = face_recognition.face_locations(image, model="hog")
							# Trích xuất encoding cho mỗi khuôn mặt
							for face_location in face_locations:
								encodings = face_recognition.face_encodings(image, [face_location])[0]
								if encodings:
									encoding = encodings[0]
									student_encodings.append(encoding)
						except Exception as e:
							print(f"\n  Lỗi khi xử lý ảnh {img_path}: {e}")
			print() # Xuống dòng sau khi xử lý xong một sinh viên
			# Lưu danh sách các encoding cho sinh viên này
			if student_encodings:
				known_encodings[student_name] = student_encodings
			else:
				print(f"  Không tìm thấy khuôn mặt nào trong thư mục của {student_name}.")

	# Lưu dữ liệu vào file
	with open(output_file, 'wb') as f:
		pickle.dump(known_encodings, f)
	print(f"Đã lưu encoding vào {output_file}")


class AttendanceGUI:
	"""
	Lớp quản lý giao diện người dùng cho hệ thống điểm danh khuôn mặt.
	Hiển thị video từ webcam, ID được phát hiện và danh sách điểm danh.
	"""
	def __init__(self, master, encodings_file='student_encodings.pkl'):
		"""
		Khởi tạo giao diện người dùng.

		Args:
			master (tk.Tk): Đối tượng cửa sổ gốc Tkinter.
			encodings_file (str): Đường dẫn đến file chứa các encoding khuôn mặt đã biết.
		"""
		self.master = master
		self.master.title("Hệ thống điểm danh khuôn mặt")
		self.master.geometry("1200x700") # Kích thước cửa sổ mặc định
		self.master.resizable(True, True) # Cho phép thay đổi kích thước cửa sổ

		self.encodings_file = encodings_file
		self.known_encodings = self._load_encodings()

		# Dictionary để lưu trạng thái điểm danh: {ID_sinh_vien: True/False}
		self.attendance_status = {name: False for name in self.known_encodings.keys()}
		self.last_detected_time = {} # Để tránh cập nhật quá nhanh

		self.detect_name_buffer = []
		self.detect_buffer_size = 10
		self.last_confirmed_name = None

		# Khởi tạo camera
		self.video_capture = cv2.VideoCapture(0)
		if not self.video_capture.isOpened():
			messagebox.showerror("Lỗi Camera", "Không thể truy cập webcam. Vui lòng kiểm tra lại.")
			self.master.destroy()
			return

		self._create_widgets()
		self._update_frame()

	def _load_encodings(self):
		"""Tải các encoding khuôn mặt đã lưu từ file."""
		if not os.path.exists(self.encodings_file):
			messagebox.showerror("Lỗi dữ liệu", f"Không tìm thấy file encoding: {self.encodings_file}\n"
											   "Vui lòng chạy chức năng huấn luyện trước.")
			return {}
		with open(self.encodings_file, 'rb') as f:
			return pickle.load(f)

	def _create_widgets(self):
		"""Tạo và sắp xếp các widget trên giao diện."""
		# Main frame chia làm 2 cột
		self.main_frame = ttk.Frame(self.master, padding="10 10 10 10")
		self.main_frame.pack(fill=tk.BOTH, expand=True)
		self.main_frame.grid_columnconfigure(0, weight=3) # Cột video rộng hơn
		self.main_frame.grid_columnconfigure(1, weight=1) # Cột thông tin
		self.main_frame.grid_rowconfigure(0, weight=1)

		# Khung bên trái: Hiển thị video
		self.video_frame = ttk.Label(self.main_frame, text="Video Webcam",
									 background="#333", foreground="white",
									 anchor="center", font=("Arial", 16))
		self.video_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

		# Khung bên phải: Chứa Detected ID và Attendance List
		self.right_frame = ttk.Frame(self.main_frame)
		self.right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
		self.right_frame.grid_rowconfigure(0, weight=0) # Detected ID box
		self.right_frame.grid_rowconfigure(1, weight=1) # Attendance List box
		self.right_frame.grid_columnconfigure(0, weight=1)

		# Khung Detected ID (trên cùng bên phải)
		self.detected_id_frame = ttk.LabelFrame(self.right_frame, text="ID được phát hiện", padding="10")
		self.detected_id_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
		self.detected_id_label = ttk.Label(self.detected_id_frame, text="Đang chờ...",
										   font=("Arial", 24, "bold"), foreground="blue")
		self.detected_id_label.pack(pady=20)

		# Khung Attendance List (dưới cùng bên phải)
		self.attendance_list_frame = ttk.LabelFrame(self.right_frame, text="Danh sách điểm danh", padding="10")
		self.attendance_list_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

		# Sử dụng Canvas và Scrollbar cho danh sách để xử lý nhiều ID
		self.canvas_attendance = tk.Canvas(self.attendance_list_frame, background="#f0f0f0")
		self.scrollbar_attendance = ttk.Scrollbar(self.attendance_list_frame, orient="vertical", command=self.canvas_attendance.yview)
		self.scrollable_frame = ttk.Frame(self.canvas_attendance)

		self.scrollable_frame.bind(
			"<Configure>",
			lambda e: self.canvas_attendance.configure(
				scrollregion=self.canvas_attendance.bbox("all")
			)
		)

		self.canvas_attendance.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
		self.canvas_attendance.configure(yscrollcommand=self.scrollbar_attendance.set)

		self.canvas_attendance.pack(side="left", fill="both", expand=True)
		self.scrollbar_attendance.pack(side="right", fill="y")

		self.student_labels = {} # Lưu trữ các Label cho từng sinh viên
		self._populate_attendance_list()

	def _populate_attendance_list(self):
		"""Điền danh sách sinh viên vào khung điểm danh."""
		for i, name in enumerate(sorted(self.known_encodings.keys())):
			status_text = "Chưa điểm danh"
			color = "red"
			if self.attendance_status[name]:
				status_text = "Đã điểm danh"
				color = "green"

			label_text = f"• {name}: {status_text}"
			student_label = ttk.Label(self.scrollable_frame, text=label_text,
									  font=("Arial", 12), foreground=color)
			student_label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
			self.student_labels[name] = student_label

	def _update_attendance_list(self, detected_name):
		"""Cập nhật trạng thái điểm danh trên giao diện."""
		if detected_name != "Unknown" and not self.attendance_status[detected_name]:
			self.attendance_status[detected_name] = True
			# Cập nhật Label của sinh viên đó
			if detected_name in self.student_labels:
				self.student_labels[detected_name].config(text=f"✓ {detected_name}: Đã điểm danh", foreground="green")

	def _update_frame(self):
		"""
		Cập nhật khung hình từ webcam và thực hiện nhận dạng khuôn mặt.
		Hàm này được gọi lặp đi lặp lại.
		"""
		ret, frame = self.video_capture.read()
		if ret:
			# Giảm kích thước khung hình để TĂNG TỐC ĐỘ PHÁT HIỆN VỊ TRÍ KHUÔN MẶT
			# Chỉ phát hiện trên ảnh nhỏ
			small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
			rgb_small_frame = small_frame[:, :, ::-1] # Chuyển đổi BGR sang RGB

			# Phát hiện vị trí khuôn mặt trong khung hình nhỏ
			face_locations_scaled = face_recognition.face_locations(rgb_small_frame, model="hog")
			
			detected_name = "Unknown" # Khởi tạo mặc định là Unknown
			current_time = time.time()

			# Với mỗi khuôn mặt được phát hiện trên ảnh nhỏ
			for (top_s, right_s, bottom_s, left_s) in face_locations_scaled:
				# Chuyển đổi tọa độ về kích thước ảnh GỐC (nhân 4 vì đã giảm 0.25)
				top = top_s * 4
				right = right_s * 4
				bottom = bottom_s * 4
				left = left_s * 4

				face_encoding = None
				try:
					# ĐIỂM QUAN TRỌNG:
					# Trích xuất encoding từ ảnh GỐC (frame) với tọa độ đã được scale lại
					face_encoding = face_recognition.face_encodings(frame, [(top, right, bottom, left)])[0]
				except Exception as e:
					print(f"Lỗi khi trích xuất encoding cho khuôn mặt: {type(e).__name__}: {e}")
					# Nếu có lỗi, bỏ qua khuôn mặt này hoặc đánh dấu là Unknown
					face_encoding = None # Đảm bảo face_encoding là None để không xử lý tiếp

				if face_encoding is not None:
					min_distances = []
					for name, encodings in self.known_encodings.items():
						if encodings:
							# Tính khoảng cách từ encoding hiện tại đến tất cả các encoding của sinh viên này
							distances = face_recognition.face_distance(encodings, face_encoding)
							min_distance = min(distances) if distances.size > 0 else float('inf')
							min_distances.append((min_distance, name))
						else:
							min_distances.append((float('inf'), name))

					# Tìm sinh viên có khoảng cách nhỏ nhất
					if min_distances:
						min_distance, match_name = min(min_distances)
						# if min_distance < 0.6:  # Ngưỡng mặc định
						# 	detected_name = match_name
						# 	# Cập nhật trạng thái điểm danh và thời gian cuối cùng phát hiện
						# 	if detected_name not in self.last_detected_time or \
						# 	   (current_time - self.last_detected_time[detected_name]) > 2: # Cập nhật sau 2 giây
						# 		self._update_attendance_list(detected_name)
						# 		self.last_detected_time[detected_name] = current_time
						# else:
						# 	detected_name = "Unknown"
						if min_distance < 0.6:
							self.detect_name_buffer.append(match_name)
							if len(self.detect_name_buffer) > self.detect_buffer_size:
								self.detect_name_buffer.pop(0)

							# Kiểm tra nếu cả 5 frame gần nhất đều là cùng 1 tên
							if len(self.detect_name_buffer) >= self.detect_buffer_size and \
							all(name == match_name for name in self.detect_name_buffer):
								self.detect_name_buffer = []
								detected_name = match_name
								if self.last_confirmed_name != detected_name:  # Tránh điểm danh lặp lại
									self._update_attendance_list(detected_name)
									self.last_detected_time[detected_name] = current_time
									self.last_confirmed_name = detected_name
						else:
							self.detect_name_buffer.append("Unknown")
							if len(self.detect_name_buffer) > self.detect_buffer_size:
								self.detect_name_buffer.pop(0)
							detected_name = "Unknown"
					else:
						detected_name = "Unknown"
				else:
					detected_name = "Unknown" # Nếu không trích xuất được encoding, coi là Unknown

				# Vẽ khung và tên sinh viên lên khung hình
				color = (0, 0, 255) if detected_name == "Unknown" else (0, 255, 0) # Đỏ cho Unknown, Xanh cho Known
				cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
				text_y = top - 10 if top - 10 > 10 else top + 20
				cv2.putText(frame, detected_name, (left + 6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


			# Cập nhật nhãn Detected ID trên GUI (có thể hiển thị người cuối cùng được detect)
			# Hoặc bạn có thể chọn hiển thị "Multiple" nếu có nhiều người và không ai là Unknown
			if face_locations_scaled: # Nếu có khuôn mặt trong khung hình
				if detected_name == "Unknown": # Cập nhật nhãn tổng thể
					self.detected_id_label.config(text="Unknown", foreground="red")
				else:
					self.detected_id_label.config(text=detected_name, foreground="green")
			else: # Không có khuôn mặt
				 self.detected_id_label.config(text="Đang chờ...", foreground="blue")


			# Chuyển đổi khung hình OpenCV sang định dạng Tkinter
			img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
			img_width, img_height = img.size
			frame_width = self.video_frame.winfo_width()
			frame_height = self.video_frame.winfo_height()

			if frame_width > 1 and frame_height > 1:
				ratio = min(frame_width / img_width, frame_height / img_height)
				new_width = int(img_width * ratio)
				new_height = int(img_height * ratio)
				img = img.resize((new_width, new_height), Image.LANCZOS)

			self.photo = ImageTk.PhotoImage(image=img)
			self.video_frame.config(image=self.photo)
			self.video_frame.image = self.photo

		self.master.after(10, self._update_frame)
		
	def on_closing(self):
		"""Xử lý khi đóng cửa sổ."""
		if messagebox.askokcancel("Thoát", "Bạn có muốn thoát ứng dụng?"):
			self.video_capture.release()
			cv2.destroyAllWindows()
			self.master.destroy()

if __name__ == "__main__":
	# Thay đổi đường dẫn đến thư mục chứa dữ liệu
	dataset_dir = r'D:\Documents\Learning\FPT\SU25\CPV\excersice\Project\Code\CPV\AI1901_face_dataset'
	output_encodings_file = 'student_encodings.pkl'

	# Bước 1: Huấn luyện mô hình (chỉ cần chạy một lần hoặc khi có dữ liệu mới)
	# Bỏ comment dòng dưới để chạy huấn luyện
	# train_faces(dataset_dir, output_encodings_file)

	# Bước 2: Chạy ứng dụng điểm danh với GUI
	root = tk.Tk()
	app = AttendanceGUI(root, encodings_file=output_encodings_file)
	root.protocol("WM_DELETE_WINDOW", app.on_closing) # Xử lý sự kiện đóng cửa sổ
	root.mainloop()

