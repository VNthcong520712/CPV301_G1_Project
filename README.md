# Instructions for Installing CMake and Python Libraries on Windows

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

## 2. Install Python 3.8

1. Download Python 3.8 from [python.org/downloads/release/python-380/](https://www.python.org/downloads/release/python-380/).
2. Run the installer. Make sure to check **Add Python to PATH** before clicking Install.

To check that Python is installed:
```sh
python --version
```
If you have many python versions, you can run code with python 3.8 with command
```sh
python3.8 --version
```
Using this format if you want to run code with python3.8

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

**Note:**  
- The `dlib` library requires CMake to build during installation, so make sure CMake is installed and added to PATH.
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