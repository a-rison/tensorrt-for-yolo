# CUDA OpenCV with Contrib Modules - Docker Setup

This project provides a Docker-based setup for building OpenCV with the `opencv_contrib` modules using CUDA (version ≥ 12.4). It is ideal for applications that require GPU acceleration for computer vision tasks.

---

## ✅ Requirements

- **CUDA Version**: >= 12.4
- **NVIDIA GPU** with drivers compatible with CUDA 12.4+
- **Docker** (version 20.10+ recommended)
- **NVIDIA Container Toolkit** for GPU support in Docker

---

## 🐳 Docker Setup

### 1. Install Docker (Ubuntu)

Follow these steps to install Docker on Ubuntu:

```bash
# Update package lists and install dependencies
sudo apt-get update
sudo apt-get install -y ca-certificates curl

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker repository to APT sources
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update and install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin



# Set up the NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install NVIDIA Container Toolkit
sudo apt update
sudo apt install -y nvidia-docker2

# Restart Docker daemon
sudo systemctl restart docker

docker compose build

docker compose up
```
### 2. Exec into the docker
```bash
docker exec -it yolo-tensorrt-container bash
```

### 3. Build with Opencv_contrib
```bash
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip
 
# Create build directory and switch into it
mkdir -p build && cd build
 
# Configure
cmake \
  -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
  -D INSTALL_PYTHON_EXAMPLES=ON \
  -D INSTALL_C_EXAMPLES=ON \
  -D WITH_TBB=ON \
  -D ENABLE_FAST_MATH=1 \
  -D CUDA_FAST_MATH=1 \
  -D WITH_CUBLAS=1 \
  -D WITH_CUDA=ON \
  -D BUILD_opencv_cudacodec=ON \
  -D WITH_CUDNN=ON \
  -D OPENCV_DNN_CUDA=ON \
  -D WITH_V4L=ON \
  -D WITH_QT=OFF \
  -D BUILD_opencv_apps=OFF \
  -D BUILD_opencv_python2=OFF \
  -D OPENCV_GENERATE_PKGCONFIG=ON \
  -D OPENCV_PC_FILE_NAME=opencv.pc \
  -D OPENCV_ENABLE_NONFREE=ON \
  -D WITH_OPENGL=OFF \
  -D WITH_GSTREAMER=ON \
  -D OPENCV_PYTHON3_INSTALL_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
  -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.x/modules \
  -D PYTHON_EXECUTABLE=$(which python3) \
  -D BUILD_EXAMPLES=ON \
  -D CUDNN_INCLUDE_DIR=/usr/include \
  -D CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so \
  -D CUDA_ARCH_BIN="8.6" \
  ..
 
# Build
make -j$(nproc)
make install
```
### 4. Prepare TRT Env
```bash
pip install tensorrt
pip install cuda-python
```
### 5. Export ONNX
```bash
pip install ultralytics

from ultralytics import YOLO
model = YOLO("yolo12n.pt")
model.export(format='onnx')
```
### 6. Generate TRT File
```bash
python export.py  -o yolo112n.onnx -e yolo12n.trt --end2end --v8 -p fp32
```
# tensorrt-for-yolo
