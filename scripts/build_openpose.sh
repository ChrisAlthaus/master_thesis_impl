#Run in the OpenPose main folder for setting up OpenPose
# -> sh ../scripts/build_openpose.sh

git submodule update --init --recursive --remote

mkdir bin
mkdir build
cd build

mkdir bin

cmake --install ../bin -D BUILD_EXAMPLES=ON  -D BUILD_PYTHON=ON  -D USE_OPENCV=ON  ..
make -j4
make install


cd python #optional?
make -j4 #optional?
