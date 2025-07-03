# Instructions for Installing CMake, C++ Development Tools, and Python Libraries on Windows

## 1. Install CMake

1. Visit the [CMake Download page](https://cmake.org/download/).
2. Download the appropriate installer for Windows (recommended: **Windows x64 Installer**).
3. Run the downloaded `.msi` file and follow the installation instructions.
4. **Note:** When prompted, choose “Add CMake to the system PATH for all users” so that you can use the `cmake` command anywhere in the Command Prompt.

To verify the installation, open Command Prompt and type:
```sh
cmake --version
```
If the CMake version is displayed, the installation was successful.

---

## 1.1. Install C++ Development Tools (Required for CMake and dlib)

To build and install `dlib` and use `cmake` on Windows, you need to install C++ development tools (Visual Studio Build Tools):

1. Go to the [Visual Studio Build Tools download page](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. Download and run the installer.
3. In the installer, select **"Desktop development with C++"**.
4. Click **Install** to begin the installation.  
   - You do not need the full Visual Studio IDE, just the Build Tools component.
5. After installation, restart your computer if prompted to update your environment variables.

**Notes:**
- Make sure that "MSVC v142 - VS 2019 C++ x64/x86 build tools" (or equivalent for your version) is selected during installation.
- Ensure that `cmake`, `cl.exe` (the C++ compiler), and build tools are available in your system PATH.

To verify, open Command Prompt and type:
```sh
cl
```
If you see information about the Microsoft C/C++ compiler, you have installed it successfully.

---

## 2. Install Python 3.8

1. Download Python 3.8 from [python.org/downloads/release/python-380/](https://www.python.org/downloads/release/python-380/).
2. Run the installer. Make sure to check **Add Python to PATH** before clicking Install.

To check that Python is installed:
```sh
python --version
```
If you have multiple Python versions, you can run code with python 3.8 using:
```sh
python3.8 --version
```
Use this format if you want to run code with python3.8 specifically.

---

## 3. (Optional but recommended) Create a Virtual Environment

```sh
python -m venv venv
venv\Scripts\activate
```

---

## 4. Install Required Python Libraries

Create a `requirements.txt` file with the following content:
```
dlib==20.0.0
opencv-python==4.10.0.84
Pillow==9.5.0
```

Then install the libraries using pip:
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

**Notes:**  
- The `dlib` library requires CMake and C++ Build Tools to build during installation, so make sure both are installed and added to your PATH.
- If you encounter errors installing `dlib`, check your CMake and Visual Studio Build Tools installation, or consider using a pre-built wheel from [https://pypi.org/project/dlib/#files](https://pypi.org/project/dlib/#files).

---

## 5. Verify the Installation

Try running the following code to check that everything is installed correctly:
```python
import cv2
import dlib
import numpy
from PIL import Image

print("All libraries are installed correctly!")
```

---

## 6. Additional Notes

- `tkinter` is usually included with Python on Windows. If it is missing, you can reinstall Python or try:
  ```
  python -m pip install tk
  ```
- If you have issues building `dlib`, refer to the official documentation or seek help from the community.

---

## 7. Contributors
This guide was contributed and maintained by the following members of the F-AI Club:
- Nguyen Thanh Cong – GitHub
- Dang Gia Duc – GitHub
- Nguyen Hoang Duy – (GitHub link not provided)
- Tran Khoi Nguyen – GitHub

We thank all contributors for their support in making AI tools and environments more accessible to the community!

---
