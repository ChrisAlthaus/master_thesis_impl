#Run in the OpenPose main folder for setting up OpenPose

git submodule update --init --recursive --remote

mkdir build
cd build

cmake -D BUILD_EXAMPLES=ON  -D BUILD_PYTHON=ON  -D USE_OPENCV=ON  ..
make -j4
cd python
make install
