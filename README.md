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
5. After installation, restaCPVT1:
- [Nguyen Thanh Cong](github.com/VNthcong520712)
- [Dang Gia Duc](https://github.com/ducgym05)
- Nguyen Hoang Duy
- [Tran Khoi Nguyen](https://github.com/KNguyenTran)

We thank all contributors for their support in making AI tools and environments more accessible to the community!

---
