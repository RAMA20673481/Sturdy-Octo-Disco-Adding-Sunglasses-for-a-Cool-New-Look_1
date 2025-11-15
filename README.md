# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look_1

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

Feel free to fork, contribute, or customize this project for your creative needs!

## Program:

 ### Workshop-1
 ### Adding Sunglasses to Your Passport Photo Using OpenCV 
 **Name:** G.Ramanujam    **Reg.No:** 212224240129

 ```
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

```
# Load the Face Image
faceImage=cv2.imread("Passport.jpg")
plt.imshow(faceImage[:,:,::-1])
plt.title("Face")
```

<img width="457" height="615" alt="Screenshot 2025-08-28 193839" src="https://github.com/user-attachments/assets/e14edb0c-153b-41ae-bdb4-900411968d52" />

```
faceImage.shape
```

<img width="155" height="35" alt="Screenshot 2025-08-28 194107" src="https://github.com/user-attachments/assets/0ff33b16-d4cc-48fe-b731-f2df6c56754c" />

```
# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glassPNG = cv2.imread('Image.png',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")
```

<img width="711" height="388" alt="Screenshot 2025-08-28 194158" src="https://github.com/user-attachments/assets/deb10d28-dcaa-4ad1-9522-8825d7676200" />

```
# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(380,80))
print("image Dimension ={}".format(glassPNG.shape))
```

<img width="294" height="36" alt="Screenshot 2025-08-28 194251" src="https://github.com/user-attachments/assets/d0da5c74-3724-4aa9-af56-609aa9b11b6d" />

```
# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]
# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');
```

<img width="884" height="136" alt="Screenshot 2025-08-28 194405" src="https://github.com/user-attachments/assets/d7ff9287-b6cd-481b-b282-2337308a03c0" />

```
# Make a copy
#faceWithGlassesNaive = resized_faceImage.copy()
faceWithGlassesNaive = faceImage.copy()

# Replace the eye region with the sunglass image
faceWithGlassesNaive[370:450, 230:610]=glassBGR

plt.imshow(faceWithGlassesNaive[...,::-1])
```

<img width="456" height="563" alt="Screenshot 2025-08-28 194454" src="https://github.com/user-attachments/assets/2eaec4e7-cdd3-4df9-acd8-25c7ad4b3749" />

```
# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3 channel image for the mask
glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))

# Make the values [0,1] since we are using arithmetic operations
glassMask = np.uint8(glassMask/255)

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Get the eye region from the face image
eyeROI= faceWithGlassesArithmetic[370:450, 230:610]

# Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI,(1-  glassMask ))

# Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR,glassMask)

# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Display the intermediate results
plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(maskedEye[...,::-1]);plt.title("Masked Eye Region")
plt.subplot(132);plt.imshow(maskedGlass[...,::-1]);plt.title("Masked Sunglass Region")
plt.subplot(133);plt.imshow(eyeRoiFinal[...,::-1]);plt.title("Augmented Eye and Sunglass")
```

<img width="889" height="126" alt="Screenshot 2025-08-28 194541" src="https://github.com/user-attachments/assets/445899f4-875c-4b3a-b594-c7fa1ef94add" />

```
# Replace the eye ROI with the output from the previous section
faceWithGlassesArithmetic[370:450, 230:610]=eyeRoiFinal

# Display the final result
plt.figure(figsize=[20,20]);
plt.subplot(121);plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image");
plt.subplot(122);plt.imshow(faceWithGlassesArithmetic[:,:,::-1]);plt.title("With Sunglasses");
```

<img width="886" height="556" alt="Screenshot 2025-08-28 194621" src="https://github.com/user-attachments/assets/d1eccac6-6b11-41d1-a6c0-fe8d218a5687" />

