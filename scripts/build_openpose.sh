#Run in the OpenPose main folder for setting up OpenPose

git submodule update --init --recursive --remote

mkdir bin
mkdir build
cd build


cmake --install ../bin -D BUILD_EXAMPLES=ON  -D BUILD_PYTHON=ON  -D USE_OPENCV=ON  ..
make -j4
make install

"""
Install the project...
-- Install configuration: "Release"
-- Installing: /usr/local/include/openpose
CMake Error at cmake_install.cmake:46 (file):
  file INSTALL cannot make directory "/usr/local/include/openpose": Read-only
  file system.

"""

cd python #optional?
make -j4 #optional?
