import cv2
import dlib
import numpy as np
import os, sys
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import threading
import time
import shutil

CAM_ID = 0 # Default camera ID, change if needed
NUM_IMAGES = 50  # Number of images to collect for each direction
DELAY = 0.1 # Delay between saves in seconds
PADD = 20 # Padding around face ROI

HOME = os.path.dirname(__file__)
sys.path.append(HOME)
print("Root directory:", HOME)
# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(HOME, r"shape_predictor_68_face_landmarks.dat"))

class FaceDataCollector:
    def __init__(self, root, saving_dir):
        self.root = root
        self.saving_dir = saving_dir
        self.root.title("Face Data Collection")
        self.root.geometry("900x700")
        
        # Create interface
        self.create_widgets()
        
        # State variables
        self.collecting = False
        self.current_direction = None
        self.directions = ["straight", "left", "right"]
        self.direction_index = 0
        self.image_count = 0
        self.cap = None
        self.face_id = ""
        self.output_dir = ""
        
        # Initialize webcam
        self.init_camera()
        
    def create_widgets(self):
        # Main frame (horizontal layout)
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left frame for status, progress, notification
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # Status frame
        status_frame = tk.Frame(left_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        self.status_label = tk.Label(status_frame, text="Status: Waiting to start", font=("Arial", 10, "bold"))
        self.status_label.pack(anchor='w')
        self.direction_label = tk.Label(status_frame, text="Direction: -", font=("Arial", 10))
        self.direction_label.pack(anchor='w')

        # Progress frame
        progress_frame = tk.Frame(left_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(progress_frame, text="Progress:").pack(anchor='w')
        self.progress_var = tk.IntVar()
        self.progress_bar = tk.Scale(progress_frame, variable=self.progress_var, from_=0, to=50, orient=tk.HORIZONTAL, length=200, showvalue=True, state=tk.DISABLED)
        self.progress_bar.pack(fill=tk.X, expand=True)

        # Notification/result frame with scrollbar
        result_frame = tk.LabelFrame(left_frame, text="Notification")
        result_frame.pack(fill=tk.BOTH, expand=True)
        result_inner_frame = tk.Frame(result_frame)
        result_inner_frame.pack(fill=tk.BOTH, expand=True)
        self.result_text = tk.Text(result_inner_frame, height=10, state=tk.DISABLED, wrap=tk.WORD)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar = tk.Scrollbar(result_inner_frame, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)

        # Control frame (put at top of right frame for better UX)
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        control_frame = tk.Frame(right_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(control_frame, text="Face ID:").pack(side=tk.LEFT, padx=(0, 10))
        self.face_id_entry = tk.Entry(control_frame, width=20)
        self.face_id_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.start_button = tk.Button(control_frame, text="Start Collection", command=self.start_collection, width=20)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = tk.Button(control_frame, text="Stop Collection", command=self.stop_collection, state=tk.DISABLED, width=20)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Video frame (camera)
        video_frame = tk.LabelFrame(right_frame, text="Camera")
        video_frame.pack(fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def init_camera(self):
        self.cap = cv2.VideoCapture(CAM_ID)
        if not self.cap.isOpened():
            self.update_result("Cannot open camera. Please check the connection.")
            return
        
        # Start updating video
        self.update_video()
    
    def update_video(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Detect face and draw result
                cv2.flip(frame, 1, frame)  # Mirror horizontally
                processed_frame, face_info = self.process_frame(frame)
                
                # Show face direction
                if face_info:
                    direction = face_info.get("direction", "-")
                    self.direction_label.config(text=f"Direction: {direction}")
                
                # Convert frame to display in Tkinter
                cv_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv_image)
                img = img.resize((640, 480), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
        
        self.root.after(10, self.update_video)
    
    def determine_face_direction(self, shape, frame_width):
        """Determine the orientation of the face based on landmarks"""
        # Get important eye and nose points
        left_eye_left_corner = shape[36]
        right_eye_right_corner = shape[45]
        nose_tip = shape[30]
        
        # Calculate vector from nose to eye points
        nose_to_left_eye = np.array([left_eye_left_corner.x - nose_tip.x, 
                                    left_eye_left_corner.y - nose_tip.y])
        nose_to_right_eye = np.array([right_eye_right_corner.x - nose_tip.x, 
                                     right_eye_right_corner.y - nose_tip.y])
        
        # Calculate vector lengths
        length_left = np.linalg.norm(nose_to_left_eye)
        length_right = np.linalg.norm(nose_to_right_eye)
        
        # Calculate ratio between sides
        ratio = length_left / (length_right + 1e-5)  # Avoid division by zero
        
        # Determine direction based on ratio
        if ratio > 1.25:  # Left eye farther -> face turns right
            return "right"
        elif ratio < 0.8:  # Right eye farther -> face turns left
            return "left"
        else:  # Ratio balanced -> face straight
            return "straight"
    
    def process_frame(self, frame):
        """Process frame to detect face and determine direction"""
        # Create a copy to avoid affecting original frame
        processed_frame = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        face_info = {}
        
        if len(faces) > 0:
            # Take the first face
            face = faces[0]
            
            # Detect landmarks
            landmarks = predictor(gray, face)
            shape = []
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                shape.append(dlib.point(x, y))
            
            # Determine face direction
            direction = self.determine_face_direction(shape, frame.shape[1])
            face_info["direction"] = direction
            
            # Draw result on frame
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            padding = PADD
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            # Draw rectangle around face
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw face direction
            cv2.putText(processed_frame, f"Direction: {direction}", 
                        (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Save ROI if collecting and matches current direction
            if self.collecting and direction == self.current_direction:
                # Compute ROI with padding
                roi = processed_frame[y1:y2, x1:x2]
                
                # Check if need to save
                current_time = time.time()
                if hasattr(self, 'last_save_time'):
                    if current_time - self.last_save_time >= DELAY:  # Save every 0.1 seconds
                        self.save_face_roi(roi)
                        self.last_save_time = current_time
                else:
                    self.last_save_time = current_time
                    self.save_face_roi(roi)
        
        return processed_frame, face_info
    
    def start_collection(self):
        """Start data collection process"""
        self.face_id = self.face_id_entry.get().strip().upper()
        if not self.face_id:
            messagebox.showerror("Error", "Please enter Face ID")
            return
        
        # Create output directory
        self.output_dir = os.path.join(self.saving_dir, f"face_dataset/{self.face_id}")
        if os.path.exists(self.output_dir):
            # Remove old directory if exists
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for each direction
        for direction in self.directions:
            os.makedirs(f"{self.output_dir}/{direction}", exist_ok=True)
        
        # Reset state
        self.collecting = True
        self.direction_index = 0
        self.image_count = 0
        self.progress_var.set(0)
        self.current_direction = self.directions[self.direction_index]
        
        # Update interface
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.face_id_entry.config(state=tk.DISABLED)
        self.progress_bar.config(state=tk.NORMAL)
        
        # Show instructions
        self.update_status()
        self.update_result(f"Start collecting data for Face ID: {self.face_id}")
        self.update_result(f"Please look {self.get_direction_text(self.current_direction)}")
    
    def get_direction_text(self, direction):
        """Convert direction name to instruction text"""
        if direction == "straight":
            return "straight"
        elif direction == "left":
            return "to the left"
        elif direction == "right":
            return "to the right"
        return direction
    
    def update_status(self):
        """Update status on the interface"""
        direction_text = self.get_direction_text(self.current_direction)
        self.status_label.config(text=f"Status: Collecting - {direction_text}")
        self.progress_var.set(self.image_count)
    
    def save_face_roi(self, roi):
        """Save cropped face image"""
        if roi.size == 0:
            return

        # Create filename
        direction_dir = f"{self.output_dir}/{self.current_direction}"
        filename = f"{direction_dir}/{self.image_count:03d}.jpg"
        print(filename)
        # Save image
        cv2.imwrite(filename, roi)

        # Update image count
        self.image_count += 1
        self.update_status()

        # Check if 50 images completed
        if self.image_count >= NUM_IMAGES:
            self.image_count = 0
            self.direction_index += 1

            if self.direction_index < len(self.directions):
                self.current_direction = self.directions[self.direction_index]
                self.update_result(f"Completed.\n===>Please look {self.get_direction_text(self.current_direction)}")
            else:
                self.stop_collection()
                self.update_result("Data collection completed!")
                self.update_result('-' * 50)  # Add 50 dashes to split samples
                messagebox.showinfo("Success", "Data collection complete!")
    
    def stop_collection(self):
        """Stop data collection process"""
        self.collecting = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.face_id_entry.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Collection stopped")
        self.update_result("Data collection stopped")
    
    def update_result(self, message):
        """Update notification/result on the interface"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.insert(tk.END, message + "\n")
        self.result_text.see(tk.END)
        self.result_text.config(state=tk.DISABLED)
    
    def on_closing(self):
        """Handle window closing event"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

def select_saving_dir(default_dir):
    saving_dir = [default_dir]  # Dùng list để thay đổi giá trị được trong hàm con

    def on_continue():
        dialog.destroy()

    def on_browse():
        dir_selected = filedialog.askdirectory(
            title="Select directory to save the dataset",
            initialdir=default_dir,
            parent=dialog
        )
        if dir_selected:
            saving_dir[0] = dir_selected
        dialog.destroy()

    dialog = tk.Toplevel()
    dialog.title("Select Saving Directory")
    dialog.geometry("430x120")
    dialog.resizable(False, False)
    label = tk.Label(
        dialog,
        text=f"Please select a directory to save the dataset.\nDefault: {default_dir}",
        wraplength=400,
        justify="left"
    )
    label.pack(padx=16, pady=(16, 8))

    button_frame = tk.Frame(dialog)
    button_frame.pack(pady=(0, 16))

    continue_btn = tk.Button(
        button_frame,
        text="Continue with default",
        width=18,
        command=on_continue
    )
    continue_btn.pack(side="left", padx=10)

    browse_btn = tk.Button(
        button_frame,
        text="Browse...",
        width=12,
        command=on_browse
    )
    browse_btn.pack(side="left")

    dialog.grab_set()
    dialog.wait_window()
    return saving_dir[0]

if __name__ == "__main__":    
    root = tk.Tk()
    root.withdraw()
    
    # messagebox.showinfo(
    #     "Select Saving Directory",
    #     f"Please select a directory to save the dataset.\nDefault is: {HOME}"
    # )
    # saving_dir = filedialog.askdirectory(
    #     title="Select directory to save the dataset",
    #     initialdir=HOME,
    #     parent=root
    # )
    saving_dir = select_saving_dir(HOME)

    root.deiconify()
    app = FaceDataCollector(root, saving_dir)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()