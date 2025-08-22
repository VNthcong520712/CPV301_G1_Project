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
import threading
import queue

def train_faces(dataset_dir, output_file='student_encodings.pkl', model="hog"):
    """
    Huấn luyện mô hình nhận dạng khuôn mặt bằng cách trích xuất các encoding từ ảnh.

    Args:     
        dataset_dir (str): Đường dẫn đến thư mục chứa dữ liệu ảnh của sinh viên.
                            Mỗi thư mục con trong dataset_dir là tên của một sinh viên,
                            và chứa các ảnh khuôn mặt của sinh viên đó.
        output_file (str): Tên file để lưu trữ các encoding đã trích xuất.
        model (str): Mô hình phát hiện khuôn mặt để sử dụng ("hog" cho CPU).
                     Lưu ý: "cnn" (GPU) đã bị loại bỏ trong phiên bản này.
    """
    if model != "hog":
        print("Cảnh báo: Chỉ hỗ trợ mô hình 'hog' (CPU). Đang chuyển sang 'hog'.")
        model = "hog"

    known_encodings = {}
    n = 0
    for student_name in os.listdir(dataset_dir):
        if "." in student_name:
            continue
        student_dir = os.path.join(dataset_dir, student_name)
        if os.path.isdir(student_dir):
            n += 1
            student_encodings = []
            print(f"{n}. Đang xử lý sinh viên: {student_name}")
            count = 0
            for root, _, files in os.walk(student_dir):
                for img_file in files:
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        count += 1
                        sys.stdout.write(f"\r   Đã xử lý {count} ảnh ({img_file})")
                        sys.stdout.flush()
                        img_path = os.path.join(root, img_file)
                        try:
                            image = face_recognition.load_image_file(img_path)
                            # Phát hiện vị trí khuôn mặt. Luôn sử dụng mô hình "hog" cho CPU.
                            face_locations = face_recognition.face_locations(image, model=model)
                            # Trích xuất encoding cho mỗi khuôn mặt
                            for face_location in face_locations:
                                encodings = face_recognition.face_encodings(image, [face_location])
                                if encodings:
                                    encoding = encodings[0]
                                    student_encodings.append(encoding)
                        except Exception as e:
                            print(f"\n   Lỗi khi xử lý ảnh {img_path}: {e}")
            print() 
                    
            if student_encodings:
                known_encodings[student_name] = student_encodings
            else:
                print(f"   Không tìm thấy khuôn mặt nào trong thư mục của {student_name}.")

    with open(output_file, 'wb') as f:
        pickle.dump(known_encodings, f)
    print(f"Đã lưu encoding vào {output_file}")

class CameraThread(threading.Thread):
    """
    Luồng riêng biệt để đọc khung hình từ webcam và đưa vào hàng đợi.
    """
    def __init__(self, video_capture, frame_queue, stop_event):
        super().__init__()
        self.video_capture = video_capture
        self.frame_queue = frame_queue
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            ret, frame = self.video_capture.read()
            if ret:
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    # Nếu hàng đợi đầy, loại bỏ khung hình cũ nhất để thêm khung hình mới
                    try:
                        self.frame_queue.get(block=False)
                        self.frame_queue.put(frame, block=False)
                    except queue.Empty: # Hàng đợi có thể trống nếu một luồng khác đã lấy
                        pass
            else:
                print("Không thể đọc khung hình từ camera.")
                break
        print("Luồng Camera đã dừng.")

class ProcessingThread(threading.Thread):
    """
    Luồng riêng biệt để xử lý nhận diện khuôn mặt từ các khung hình trong hàng đợi.
    """
    def __init__(self, frame_queue, known_encodings, attendance_status, last_detected_time,
                 detect_name_buffer, detect_buffer_size, last_confirmed_name,
                 detected_id_label_update_callback, update_attendance_list_callback, stop_event,
                 detection_model="hog"): # Tham số detection_model, mặc định là "hog"
        super().__init__()
        self.frame_queue = frame_queue
        self.known_encodings = known_encodings
        self.attendance_status = attendance_status
        self.last_detected_time = last_detected_time
        self.detect_name_buffer = detect_name_buffer
        self.detect_buffer_size = detect_buffer_size
        self.last_confirmed_name = last_confirmed_name
        self.detected_id_label_update_callback = detected_id_label_update_callback
        self.update_attendance_list_callback = update_attendance_list_callback
        self.stop_event = stop_event
        self.frame_to_display = None
        self.detection_model = detection_model # Lưu trữ mô hình phát hiện

    def run(self):
        while not self.stop_event.is_set():
            try:
                start_total_time = time.perf_counter() 

                start_queue_get_time = time.perf_counter()
                frame = self.frame_queue.get(timeout=0.1)
                end_queue_get_time = time.perf_counter()
                # print(f"Thời gian lấy khung hình từ hàng đợi: {end_queue_get_time - start_queue_get_time:.4f} giây")

                # Giảm kích thước khung hình để tăng tốc độ phát hiện khuôn mặt.
                # Sử dụng fx=0.25, fy=0.25 để giảm kích thước ảnh xuống 1/4.
                start_resize_time = time.perf_counter()
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1] # Chuyển đổi BGR sang RGB
                end_resize_time = time.perf_counter()
                # print(f"Thời gian thay đổi kích thước và chuyển đổi màu: {end_resize_time - start_resize_time:.4f} giây")

                start_face_detection_time = time.perf_counter()
                # Sử dụng self.detection_model (luôn là "hog" trong phiên bản này)
                face_locations_scaled = face_recognition.face_locations(rgb_small_frame, model=self.detection_model)
                end_face_detection_time = time.perf_counter()
                # print(f"Thời gian lấy vị trí khuôn mặt ({self.detection_model}): {end_face_detection_time - start_face_detection_time:.4f} giây")
                
                detected_name = "Unknown" 
                current_time = time.time()
                
                display_frame = frame.copy()

                for (top_s, right_s, bottom_s, left_s) in face_locations_scaled:
                    top = top_s * 4 # Phục hồi tọa độ về kích thước gốc
                    right = right_s * 4
                    bottom = bottom_s * 4
                    left = left_s * 4

                    face_encoding = None
                    start_encoding_time = time.perf_counter()
                    try:
                        # Trích xuất encoding từ khung hình gốc với tọa độ đã phục hồi
                        face_encoding = face_recognition.face_encodings(frame, [(top, right, bottom, left)])[0]
                    except IndexError: # Xử lý trường hợp không tìm thấy encoding cho khuôn mặt
                        face_encoding = None
                    except Exception as e:
                        print(f"Lỗi khi trích xuất encoding cho khuôn mặt: {type(e).__name__}: {e}")
                        face_encoding = None
                    end_encoding_time = time.perf_counter()
                    # print(f"   Thời gian trích xuất encoding: {end_encoding_time - start_encoding_time:.4f} giây")

                    if face_encoding is not None:
                        start_comparison_matching_time = time.perf_counter()
                        min_distances = []
                        for name, encodings in self.known_encodings.items():
                            if encodings:
                                distances = face_recognition.face_distance(encodings, face_encoding)
                                min_distance = min(distances) if distances.size > 0 else float('inf')
                                min_distances.append((min_distance, name))
                            else:
                                min_distances.append((float('inf'), name))

                        if min_distances:
                            min_distance, match_name = min(min_distances)
                            if min_distance < 0.6: # Ngưỡng nhận diện, có thể điều chỉnh
                                self.detect_name_buffer.append(match_name)
                                if len(self.detect_name_buffer) > self.detect_buffer_size:
                                    self.detect_name_buffer.pop(0)

                                if len(self.detect_name_buffer) >= self.detect_buffer_size and \
                                all(name == match_name for name in self.detect_name_buffer):
                                    self.detect_name_buffer = [] # Xóa buffer sau khi xác nhận
                                    detected_name = match_name
                                    start_attendance_evaluation_time = time.perf_counter()
                                    if self.last_confirmed_name != detected_name: # Chỉ cập nhật khi tên thay đổi
                                        self.update_attendance_list_callback(detected_name)
                                        self.last_detected_time[detected_name] = current_time
                                        self.last_confirmed_name = detected_name
                                    end_attendance_evaluation_time = time.perf_counter()
                                    # print(f"   Thời gian xử lý đánh giá có mặt: {end_attendance_evaluation_time - start_attendance_evaluation_time:.4f} giây")
                                else:
                                    # Nếu buffer chưa đủ hoặc không đồng nhất, vẫn coi là Unknown cho đến khi đủ điều kiện
                                    detected_name = "Unknown" 
                            else:
                                detected_name = "Unknown"
                        else:
                            detected_name = "Unknown"
                        end_comparison_matching_time = time.perf_counter()
                        # print(f"   Thời gian so sánh encoding và matching: {end_comparison_matching_time - start_comparison_matching_time:.4f} giây")
                    else:
                        detected_name = "Unknown"

                    start_drawing_time = time.perf_counter()
                    color = (0, 0, 255) if detected_name == "Unknown" else (0, 255, 0)
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    text_y = top - 10 if top - 10 > 10 else top + 20
                    cv2.putText(display_frame, detected_name, (left + 6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    end_drawing_time = time.perf_counter()
                    # print(f"   Thời gian vẽ khung và chữ: {end_drawing_time - start_drawing_time:.4f} giây")

                start_gui_update_time = time.perf_counter()
                if face_locations_scaled:
                    if detected_name == "Unknown":
                        self.detected_id_label_update_callback("Unknown", "red")
                    else:
                        self.detected_id_label_update_callback(detected_name, "green")
                else:
                    self.detected_id_label_update_callback("Đang chờ...", "blue")
                end_gui_update_time = time.perf_counter()
                # print(f"Thời gian cập nhật GUI (callback): {end_gui_update_time - start_gui_update_time:.4f} giây")

                self.frame_to_display = display_frame 
                
                end_total_time = time.perf_counter() 
                # print(f"Tổng thời gian xử lý khung hình: {end_total_time - start_total_time:.4f} giây\n")

            except queue.Empty:
                pass 
            except Exception as e:
                print(f"Lỗi trong luồng xử lý: {e}")
        print("Luồng Processing đã dừng.")


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
        self.master.title("Hệ thống điểm danh khuôn mặt (CPU Only)")
        self.master.geometry("1200x700") 
        self.master.resizable(True, True) 

        self.encodings_file = encodings_file
        self.known_encodings = self._load_encodings()

        self.attendance_status = {name: False for name in self.known_encodings.keys()}
        self.last_detected_time = {}

        self.detect_name_buffer = []
        self.detect_buffer_size = 10 # Số khung hình liên tiếp cần xác nhận để điểm danh
        self.last_confirmed_name = None

        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            messagebox.showerror("Lỗi Camera", "Không thể truy cập webcam. Vui lòng kiểm tra lại.")
            self.master.destroy()
            return
        
        # Mô hình phát hiện khuôn mặt được cố định là "hog" (CPU)
        self.detection_model = "hog" 
        self.processing_thread = None # Khởi tạo None để kiểm tra sau

        self.frame_queue = queue.Queue(maxsize=15) 
        self.stop_event = threading.Event()

        self.camera_thread = CameraThread(self.video_capture, self.frame_queue, self.stop_event)
        self.camera_thread.daemon = True
        self.camera_thread.start()

        self._create_widgets()
        self._start_processing_thread() # Bắt đầu luồng xử lý sau khi tạo widget
        self._update_frame_display() 

    def _load_encodings(self):
        """Tải các encoding khuôn mặt đã lưu từ file."""
        if not os.path.exists(self.encodings_file):
            messagebox.showerror("Lỗi dữ liệu", f"Không tìm thấy file encoding: {self.encodings_file}\n"
                                               "Vui lòng chạy chức năng huấn luyện trước.")
            return {}
        with open(self.encodings_file, 'rb') as f:
            return pickle.load(f)

    def _start_processing_thread(self):
        """Khởi tạo và bắt đầu luồng xử lý khuôn mặt."""
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.stop_event.set()
            self.processing_thread.join() # Chờ luồng cũ dừng lại

        self.processing_thread = ProcessingThread(
            self.frame_queue,
            self.known_encodings,
            self.attendance_status,
            self.last_detected_time,
            self.detect_name_buffer,
            self.detect_buffer_size,
            self.last_confirmed_name,
            self._update_detected_id_label,
            self._update_attendance_list,
            self.stop_event,
            detection_model=self.detection_model # Truyền mô hình đã chọn ("hog")
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
        print(f"Luồng xử lý đã bắt đầu với mô hình: {self.detection_model}")

    # Loại bỏ hàm _on_model_change vì không còn lựa chọn mô hình
    # def _on_model_change(self, *args):
    #     """Xử lý khi người dùng thay đổi lựa chọn mô hình."""
    #     selected_model = self.detection_model.get()
    #     print(f"Đã thay đổi mô hình phát hiện sang: {selected_model}")
    #     self._start_processing_thread()

    def _create_widgets(self):
        """Tạo và sắp xếp các widget trên giao diện."""
        self.main_frame = ttk.Frame(self.master, padding="10 10 10 10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.grid_columnconfigure(0, weight=3)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.video_frame = ttk.Label(self.main_frame, text="Video Webcam",
                                     background="#333", foreground="white",
                                     anchor="center", font=("Arial", 16))
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.right_frame.grid_rowconfigure(0, weight=0) # Cho khung ID
        # self.right_frame.grid_rowconfigure(1, weight=0) # Loại bỏ hàng cho lựa chọn mô hình
        self.right_frame.grid_rowconfigure(1, weight=1) # Cho danh sách điểm danh (thay đổi từ row 2 thành row 1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        # Khung ID được phát hiện
        self.detected_id_frame = ttk.LabelFrame(self.right_frame, text="ID được phát hiện", padding="10")
        self.detected_id_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.detected_id_label = ttk.Label(self.detected_id_frame, text="Đang chờ...",
                                             font=("Arial", 24, "bold"), foreground="blue")
        self.detected_id_label.pack(pady=20)

        # Loại bỏ khung lựa chọn mô hình vì chỉ sử dụng CPU (HOG)
        # self.model_selection_frame = ttk.LabelFrame(self.right_frame, text="Chọn mô hình phát hiện", padding="10")
        # self.model_selection_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        # self.cpu_radio = ttk.Radiobutton(self.model_selection_frame, text="CPU (HOG)",
        #                                  variable=self.detection_model, value="hog",
        #                                  command=self._on_model_change)
        # self.cpu_radio.pack(anchor="w", pady=5)
        # self.gpu_radio = ttk.Radiobutton(self.model_selection_frame, text="GPU (CNN)",
        #                                  variable=self.detection_model, value="cnn",
        #                                  command=self._on_model_change)
        # self.gpu_radio.pack(anchor="w", pady=5)


        self.attendance_list_frame = ttk.LabelFrame(self.right_frame, text="Danh sách điểm danh", padding="10")
        self.attendance_list_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5) # Thay đổi row thành 1

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

        self.student_labels = {} 
        self._populate_attendance_list()

    def _populate_attendance_list(self):
        """Điền danh sách sinh viên vào khung điểm danh."""
        # Xóa các label cũ trước khi điền lại
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.student_labels = {}

        for i, name in enumerate(sorted(self.known_encodings.keys())):
            status_text = "Chưa điểm danh"
            color = "red"
            if self.attendance_status.get(name, False): # Sử dụng .get để tránh lỗi nếu tên không có trong attendance_status
                status_text = "Đã điểm danh"
                color = "green"

            label_text = f"• {name}: {status_text}"
            student_label = ttk.Label(self.scrollable_frame, text=label_text,
                                     font=("Arial", 12), foreground=color)
            student_label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
            self.student_labels[name] = student_label

    def _update_attendance_list(self, detected_name):
        """Cập nhật trạng thái điểm danh trên giao diện. (Được gọi từ luồng xử lý)"""
        self.master.after(0, self.__update_attendance_list_gui, detected_name)

    def __update_attendance_list_gui(self, detected_name):
        """Hàm nội bộ để cập nhật GUI cho danh sách điểm danh."""
        if detected_name != "Unknown" and not self.attendance_status.get(detected_name, False):
            self.attendance_status[detected_name] = True
            if detected_name in self.student_labels:
                self.student_labels[detected_name].config(text=f"✓ {detected_name}: Đã điểm danh", foreground="green")

    def _update_detected_id_label(self, text, color):
        """Cập nhật nhãn ID được phát hiện trên giao diện. (Được gọi từ luồng xử lý)"""
        self.master.after(0, self.__update_detected_id_label_gui, text, color)

    def __update_detected_id_label_gui(self, text, color):
        """Hàm nội bộ để cập nhật GUI cho nhãn ID."""
        self.detected_id_label.config(text=text, foreground=color)

    def _update_frame_display(self):
        """
        Chỉ cập nhật khung hình hiển thị từ luồng xử lý.
        Hàm này được gọi lặp đi lặp lại.
        """
        if self.processing_thread and self.processing_thread.frame_to_display is not None:
            frame = self.processing_thread.frame_to_display
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

        self.master.after(10, self._update_frame_display)
        
    def on_closing(self):
        """Xử lý khi đóng cửa sổ."""
        if messagebox.askokcancel("Thoát", "Bạn có muốn thoát ứng dụng?"):
            self.stop_event.set() 
            if self.camera_thread and self.camera_thread.is_alive():
                self.camera_thread.join() 
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join() 
            self.video_capture.release()
            cv2.destroyAllWindows()
            self.master.destroy()

if __name__ == "__main__":
    # Thay đổi đường dẫn đến thư mục dữ liệu của bạn
    dataset_dir = r'D:\Documents\Learning\FPT\SU25\CPV\excersice\Project\Code\CPV\AI1901_face_dataset'
    output_encodings_file = 'student_encodings.pkl'

    # Bước 1: Huấn luyện mô hình (chỉ cần chạy một lần hoặc khi có dữ liệu mới)
    # Luôn sử dụng model="hog" để chạy trên CPU.
    # Bỏ comment dòng dưới đây nếu bạn cần huấn luyện lại dữ liệu.
    # train_faces(dataset_dir, output_encodings_file, model="hog") 

    # Bước 2: Chạy ứng dụng điểm danh với GUI
    root = tk.Tk()
    app = AttendanceGUI(root, encodings_file=output_encodings_file)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
