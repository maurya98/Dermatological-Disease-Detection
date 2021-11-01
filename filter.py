import cv2 
import matplotlib.pyplot as plt 
import numpy as np
# Reads the image 
img = cv2.imread('/root/Desktop/majors/dataset/train/Acne/07Acne081101.jpg')
img2= cv2.imread('/root/Desktop/majors/dataset/train/Acne/07AcnePittedScars.jpg')
# image by using cv2color 
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  


# Convert to YCrCb color space 
#img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) 
#array=[img]
#print(array)
#plt.imshow(img)   

# Converts to HSV color space 

img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) 
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2LAB)
array=[img]
print(array)
plt.imshow(img)   
plt.imshow(img2)

# Converts to LAB color space 
#img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#plt.imshow(img)   

# as EdgeMap 
laplacian = cv2.Laplacian(img, cv2.CV_64F) 
laplacian_2 =cv2.Laplacian(img2, cv2.CV_64F)
array=[laplacian]
array_2=[laplacian_2]
print(array)
print(array_2)

# Shows the image
plt.imshow(laplacian) 
plt.imshow(laplacian_2)

# Spectral map of image 
plt.imshow(img, cmap ='nipy_spectral')
plt.imshow(img2,cmap='nipy_spectral')
#plt.imshow(img)   

kernel_sharpening = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
# applying the sharpening kernel to the input image & displaying it.
sharpened = cv2.filter2D(img, -1, kernel_sharpening)
plt.imshow(sharpened)