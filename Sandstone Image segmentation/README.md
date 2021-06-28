# Sandstone Image Segmentation using Random Forest

## Overview
The dataset we are working on are the Micro CT scansor X-ray microscope scans of a sandstone. In the image there are bright regions are heavy material, the Gray regions are Cores, the textured regions are some sort of a Clay, and the dark black region is air etc given in the images. So by this machine learning model we segment all the regions.

When we have less number of data then this Random Forest technique is more accurate then other algorithm for image segmentation.
So here in this project we choose images from dataset and doing feature engineering on that image using verious filter like Gabor, Scharr, Robert, Sobel, Canny, Gaussian, Median, Prewitt, Variance etc and extracting feature from that and insert it in new dataframe for training the model.
after that split the data into training and testing and then we train our Random Forest model with training dataset.

## Model Accuracy on Sandstone dataset
  98.13%

## Screenshots

<p align="center">
  <img src="https://github.com/Lalit78716/Image-segmentation-Projects/blob/main/Sandstone%20Image%20segmentation/Screenshots/Screenshot%20(506).png" width="350" title=" Result">
  </p>
  
### If i test it on other image it gives this result

<p align="center">
  <img src="https://github.com/Lalit78716/Image-segmentation-Projects/blob/main/Sandstone%20Image%20segmentation/Screenshots/Screenshot%20(507).png" width="350" title=" Result">
  </p>
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
## Reference:-  
## Project is inspired from  [Digital Sreeni]: https://www.youtube.com/channel/UC34rW-HtPJulxr5wp2Xa04w
## Special Thanks to @bnsreenu sir who taught us ML/DL for microscopic level
  
