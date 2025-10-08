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

# Part 1: Data preparation (Training)
def train_faces(dataset_dir, output_file='student_encodings.pkl'):
	"""
	Train the face recognition model by extracting encodings from images.

	Args:
		dataset_dir (str): Path to the directory containing student image data.
						   Each subfolder in dataset_dir represents a student's name
						   and contains that student's face images.
		output_file (str): Filename to save extracted encodings.
	"""
	known_encodings = {}  # key: student name, value: list of encodings
	n = 0
	# Iterate through each student's folder
	for student_name in os.listdir(dataset_dir):
		# Skip non-folder files
		if "." in student_name:
			continue
		student_dir = os.path.join(dataset_dir, student_name)
		if os.path.isdir(student_dir):
			n += 1
			student_encodings = []
			print(f"{n}. Processing student: {student_name}")
			count = 0
			# Iterate through subfolders (if any) or directly images
			for root, _, files in os.walk(student_dir):
				for img_file in files:
					# Only process common image file types
					if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
						count += 1
						sys.stdout.write(f"\r  Processed {count} images ({img_file})")
						sys.stdout.flush()
						# time.sleep(0.01)  # Simulate processing delay (optional)
						img_path = os.path.join(root, img_file)
						try:
							# Load image
							image = face_recognition.load_image_file(img_path)
							# Detect face locations
							face_locations = face_recognition.face_locations(image, model="hog")
							# Extract encodings for each detected face
							for face_location in face_locations:
								encodings = face_recognition.face_encodings(image, [face_location])[0]
								if encodings:
									encoding = encodings[0]
									student_encodings.append(encoding)
						except Exception as e:
							print(f"\n  Error processing image {img_path}: {e}")
			print() # New line after processing one student
			# Save the encoding list for this student
			if student_encodings:
				known_encodings[student_name] = student_encodings
			else:
				print(f"  No face found in folder for {student_name}.")

	# Save data to file
	with open(output_file, 'wb') as f:
		pickle.dump(known_encodings, f)
	print(f"Encodings saved to {output_file}")


class AttendanceGUI:
	"""
	GUI class for the face recognition attendance system.
	Displays webcam video, detected ID, and attendance list.
	"""
	def __init__(self, master, encodings_file='student_encodings.pkl'):
		"""
		Initialize the user interface.

		Args:
			master (tk.Tk): Tkinter root window object.
			encodings_file (str): Path to the file containing known face encodings.
		"""
		self.master = master
		self.master.title("Face Attendance System")
		self.master.geometry("1200x700") # Default window size
		self.master.resizable(True, True) # Allow resizing

		self.encodings_file = encodings_file
		self.known_encodings = self._load_encodings()

		# Dictionary to store attendance status: {student_ID: True/False}
		self.attendance_status = {name: False for name in self.known_encodings.keys()}
		self.last_detected_time = {} # To prevent too frequent updates

		self.detect_name_buffer = []
		self.detect_buffer_size = 10
		self.last_confirmed_name = None

		# Initialize camera
		self.video_capture = cv2.VideoCapture(0)
		if not self.video_capture.isOpened():
			messagebox.showerror("Camera Error", "Cannot access webcam. Please check your device.")
			self.master.destroy()
			return

		self._create_widgets()
		self._update_frame()

	def _load_encodings(self):
		"""Load saved face encodings from file."""
		if not os.path.exists(self.encodings_file):
			messagebox.showerror("Data Error", f"Encoding file not found: {self.encodings_file}\n"
											   "Please run the training function first.")
			return {}
		with open(self.encodings_file, 'rb') as f:
			return pickle.load(f)

	def _create_widgets(self):
		"""Create and arrange UI widgets."""
		# Main frame split into 2 columns
		self.main_frame = ttk.Frame(self.master, padding="10 10 10 10")
		self.main_frame.pack(fill=tk.BOTH, expand=True)
		self.main_frame.grid_columnconfigure(0, weight=3) # Wider column for video
		self.main_frame.grid_columnconfigure(1, weight=1) # Narrower for info
		self.main_frame.grid_rowconfigure(0, weight=1)

		# Left frame: display webcam video
		self.video_frame = ttk.Label(self.main_frame, text="Webcam Video",
									 background="#333", foreground="white",
									 anchor="center", font=("Arial", 16))
		self.video_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

		# Right frame: contains Detected ID and Attendance List
		self.right_frame = ttk.Frame(self.main_frame)
		self.right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
		self.right_frame.grid_rowconfigure(0, weight=0) # Detected ID box
		self.right_frame.grid_rowconfigure(1, weight=1) # Attendance List box
		self.right_frame.grid_columnconfigure(0, weight=1)

		# Detected ID section (top right)
		self.detected_id_frame = ttk.LabelFrame(self.right_frame, text="Detected ID", padding="10")
		self.detected_id_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
		self.detected_id_label = ttk.Label(self.detected_id_frame, text="Waiting...",
										   font=("Arial", 24, "bold"), foreground="blue")
		self.detected_id_label.pack(pady=20)

		# Attendance list section (bottom right)
		self.attendance_list_frame = ttk.LabelFrame(self.right_frame, text="Attendance List", padding="10")
		self.attendance_list_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

		# Use Canvas and Scrollbar for long lists
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

		self.student_labels = {} # Store Labels for each student
		self._populate_attendance_list()

	def _populate_attendance_list(self):
		"""Fill the attendance list on the interface."""
		for i, name in enumerate(sorted(self.known_encodings.keys())):
			status_text = "Not Checked In"
			color = "red"
			if self.attendance_status[name]:
				status_text = "Checked In"
				color = "green"

			label_text = f"• {name}: {status_text}"
			student_label = ttk.Label(self.scrollable_frame, text=label_text,
									  font=("Arial", 12), foreground=color)
			student_label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
			self.student_labels[name] = student_label

	def _update_attendance_list(self, detected_name):
		"""Update attendance status on the GUI."""
		if detected_name != "Unknown" and not self.attendance_status[detected_name]:
			self.attendance_status[detected_name] = True
			# Update that student's label
			if detected_name in self.student_labels:
				self.student_labels[detected_name].config(text=f"✓ {detected_name}: Checked In", foreground="green")

	def _update_frame(self):
		"""
		Update webcam frame and perform face recognition.
		This function is called repeatedly.
		"""
		ret, frame = self.video_capture.read()
		if ret:
			# Reduce frame size to INCREASE FACE DETECTION SPEED
			# Detect only on smaller image
			small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
			rgb_small_frame = small_frame[:, :, ::-1] # Convert BGR to RGB

			# Detect face locations in the small frame
			face_locations_scaled = face_recognition.face_locations(rgb_small_frame, model="hog")
			
			detected_name = "Unknown" # Default to Unknown
			current_time = time.time()

			# For each detected face in the small frame
			for (top_s, right_s, bottom_s, left_s) in face_locations_scaled:
				# Scale back coordinates to ORIGINAL image size (×4 since reduced to 0.25)
				top = top_s * 4
				right = right_s * 4
				bottom = bottom_s * 4
				left = left_s * 4

				face_encoding = None
				try:
					# IMPORTANT:
					# Extract encoding from ORIGINAL frame using rescaled coordinates
					face_encoding = face_recognition.face_encodings(frame, [(top, right, bottom, left)])[0]
				except Exception as e:
					print(f"Error extracting face encoding: {type(e).__name__}: {e}")
					# If error, skip this face or mark as Unknown
					face_encoding = None # Ensure None for skipping

				if face_encoding is not None:
					min_distances = []
					for name, encodings in self.known_encodings.items():
						if encodings:
							# Compute distance from current encoding to all encodings of this student
							distances = face_recognition.face_distance(encodings, face_encoding)
							min_distance = min(distances) if distances.size > 0 else float('inf')
							min_distances.append((min_distance, name))
						else:
							min_distances.append((float('inf'), name))

					# Find the student with smallest distance
					if min_distances:
						min_distance, match_name = min(min_distances)
						if min_distance < 0.6:
							self.detect_name_buffer.append(match_name)
							if len(self.detect_name_buffer) > self.detect_buffer_size:
								self.detect_name_buffer.pop(0)

							# Check if all recent frames are the same name
							if len(self.detect_name_buffer) >= self.detect_buffer_size and \
							all(name == match_name for name in self.detect_name_buffer):
								self.detect_name_buffer = []
								detected_name = match_name
								if self.last_confirmed_name != detected_name:  # Prevent repeated check-ins
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
					detected_name = "Unknown" # If failed to extract encoding

				# Draw rectangle and name on frame
				color = (0, 0, 255) if detected_name == "Unknown" else (0, 255, 0) # Red for Unknown, Green for Known
				cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
				text_y = top - 10 if top - 10 > 10 else top + 20
				cv2.putText(frame, detected_name, (left + 6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


			# Update Detected ID label on GUI
			# Optionally display "Multiple" if multiple known faces and none Unknown
			if face_locations_scaled: # If faces detected
				if detected_name == "Unknown":
					self.detected_id_label.config(text="Unknown", foreground="red")
				else:
					self.detected_id_label.config(text=detected_name, foreground="green")
			else: # No face
				 self.detected_id_label.config(text="Waiting...", foreground="blue")


			# Convert OpenCV frame to Tkinter-compatible format
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
		"""Handle window closing event."""
		if messagebox.askokcancel("Exit", "Do you want to exit the application?"):
			self.video_capture.release()
			cv2.destroyAllWindows()
			self.master.destroy()

if __name__ == "__main__":
	# Change this path to your dataset directory
	dataset_dir = r'D:\Documents\Learning\FPT\SU25\CPV\excersice\Project\Code\CPV\AI1901_face_dataset'
	output_encodings_file = 'student_encodings.pkl'

	# Step 1: Train the model (only once or when new data is added)
	# Uncomment the line below to run training
	# train_faces(dataset_dir, output_encodings_file)

	# Step 2: Run the attendance GUI
	root = tk.Tk()
	app = AttendanceGUI(root, encodings_file=output_encodings_file)
	root.protocol("WM_DELETE_WINDOW", app.on_closing) # Handle window close event
	root.mainloop()
