import numpy as np
from PIL import Image
from ISR.models import RDN
import matplotlib.pyplot as plt
from skimage import io


path = './usage/tar_images/IMG_1006.jpg'
img = Image.open(path)
lr_img = np.array(img)
rdn = RDN(weights='noise-cancel')
sr_img = rdn.predict(lr_img)
lr_img = np.array(img.resize((1024, 1024)))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(lr_img, cmap='gray')
ax2.imshow(sr_img, cmap='gray')
plt.show()
io.imsave(path, lr_img)