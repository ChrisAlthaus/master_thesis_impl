 
from PIL import Image
import numpy as np
import detectron2.data.transforms as T

img = np.array(Image.open('/home/althausc/nfs/data/styleimgs_test/out/09/15_13-30-19/000000000632_062187.jpg'))

for i in range(10):
    a_rb = T.RandomBrightness(intensity_min=0.75, intensity_max=1.25)
    t = a_rb.get_transform(img)
    Image.fromarray(t.apply_image(img)).save('.images/a_rb%d.jpg'%i)

for i in range(10):
    a_rc = T.RandomContrast(intensity_min=0.76, intensity_max=1.25)
    t = a_rc.get_transform(img)
    Image.fromarray(t.apply_image(img)).save('.images/a_rc%d.jpg'%i)
    
for i in range(10):
    a_rs = T.RandomSaturation(intensity_min=0.75, intensity_max=1.25)
    t = a_rs.get_transform(img)
    Image.fromarray(t.apply_image(img)).save('.images/a_rs%d.jpg'%i)

for i in range(10):
    a_rr = T.RandomRotation([-15,15])
    t = a_rr.get_transform(img)
    Image.fromarray(t.apply_image(img)).save('.images/a_rr%d.jpg'%i)

