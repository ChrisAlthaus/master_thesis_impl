import os
try:
    user_paths = os.environ['PYTHONPATH'] #.split(os.pathsep)
except KeyError:
    user_paths = []
    print("Keyerror")

print(user_paths)
from deeplab import model