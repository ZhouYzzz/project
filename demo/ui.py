import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

plt.figure(figsize=(4,4))
img = mpimg.imread('tmp.jpg')
plt.imshow(img)

plt.show()

time.sleep(2)
img = mpimg.imread('person_im.jpg')
plt.imshow(img)
