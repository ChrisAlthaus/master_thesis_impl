#Loads only the deeplab model from the tensorflow model repository
#Please run from the root directory
mkdir deeplab
cd deeplab
#git init
#git config core.sparseCheckout true
#git remote add -f origin https://github.com/tensorflow/models.git
#echo "research/deeplab/*" > .git/info/sparse-checkout
#git checkout master

#pip install --user svn
echo "SVN checkout might take up to 30 seconds..."
svn checkout https://github.com/tensorflow/models/trunk/research/deeplab
echo "checkout done."

echo "SVN checkout might take up to 30 seconds..."
svn checkout https://github.com/tensorflow/models/trunk/research/slim
echo "checkout done."

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
print("Printing python path:")
python -c "import sys; print(sys.path)"
