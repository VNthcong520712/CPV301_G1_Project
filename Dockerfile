FROM python:3.8-slim

# Tạo thư mục làm việc
WORKDIR /app

COPY . .

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && \
    apt-get install -y cmake build-essential \
    libgtk2.0-dev libcanberra-gtk* libsm6 libxext6 libxrender-dev \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements và cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Command mặc định, thay đổi nếu bạn có file python cụ thể
# CMD ["python", "Adding_dataset/face_id_collection.py"]