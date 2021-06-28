# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 11:21:29 2021

@author: Lalit Mali
"""
import pandas as pd
import cv2
import numpy as np
from skimage.filters import roberts,sobel,scharr,prewitt
from scipy import ndimage as nd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
img = cv2.imread('Sandstone_Versa0000.tif',1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

df = pd.DataFrame()

#feature no. 1 is our original pixel values of image
img2=img.reshape(-1)
df['Original Image']=img2

# add other features
#first set Gabor features

num=1
kernels=[]
for theta in range(2):
    theta=theta/4. * np.pi
    for sigma in (3,5):
        for lamda in np.arange(0,np.pi,np.pi /4.):
            for gamma in (0.05,0.5):
                gabor_label='Gabor'+str(num)
                ksize=5
                kernel=cv2.getGaborKernel((ksize,ksize),sigma,theta,lamda,0,ktype=cv2.CV_32F)
                kernels.append(kernel)
                
                fimg=cv2.filter2D(img2,cv2.CV_8UC3,kernel)
                filtered_img=fimg.reshape(-1)
                df[gabor_label]=filtered_img
                num+=1

# canny edge

edges=cv2.Canny(img,100,200)
edges1=edges.reshape(-1)
df['Canny Edge']=edges1
#cv2.imshow("canny",edges)

# adding robert,sobel, scharr, prewitt
edge_roberts=roberts(img)
edge_roberts1=edge_roberts.reshape(-1)
df['Roberts']=edge_roberts1

edge_sobel=sobel(img)
edge_sobel1=edge_sobel.reshape(-1)
df['Sobel']=edge_sobel1

edge_scharr=scharr(img)
edge_scharr1=edge_scharr.reshape(-1)
df['Scharr']=edge_scharr1

edge_prewitt=prewitt(img)
edge_prewitt1=edge_prewitt.reshape(-1)
df['Prewitt']=edge_prewitt1

# Gaussian with sigma=3
gaussian_img=nd.gaussian_filter(img,sigma=3)
gaussian_img1=gaussian_img.reshape(-1)
df['Gaussian s3']=gaussian_img1

# Gaussian with sigma=7
gaussian_img2=nd.gaussian_filter(img,sigma=7)
gaussian_img3=gaussian_img2.reshape(-1)
df['Gaussian s7']=gaussian_img3

#median with sigma=3
median_img=nd.median_filter(img,size=3)
median_img1=median_img.reshape(-1)
df['Median s3']=median_img1

#variance with size=3
#np.var is finding the variance of an image
#variance_img=nd.generic_filter(img,np.var,size=3)
#variance_img1=variance_img.reshape(-1)
#df['Variance s3']=variance_img1

labeled_img=cv2.imread('Train_label_masks/Sandstone_Versa0000.tif')
labeled_img=cv2.cvtColor(labeled_img,cv2.COLOR_BGR2GRAY)
labeled_img1=labeled_img.reshape(-1)
df['Labels']=labeled_img1

#print(df.head())

#dependent variable
Y=df['Labels'].values

#independent variables
X=df.drop(labels=['Labels'],axis=1)

#split data into train and test
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.40,random_state=20)

#import ML algo and train the data
model=RandomForestClassifier(n_estimators=10,random_state=42)
model.fit(X_train,Y_train)

pred_test=model.predict(X_test)

from sklearn import metrics
print('Accuracy : ',metrics.accuracy_score(Y_test,pred_test))

#importances=list(model.feature_importances_)

features_list=list(X.columns)
feature_imp=pd.Series(model.feature_importances_,index=features_list).sort_values(
    ascending=False)
print(feature_imp)
#you can top 5 features from that


filename='sandstone_model'
pickle.dump(model,open(filename,'wb'))

# if i want to work with this pickle model then
load_model=pickle.load(open(filename,'rb'))
result=load_model.predict(X)

segmented=result.reshape((img.shape))
plt.imshow(segmented,cmap='jet')
plt.imsave('segmented_rock.jpg',segmented,cmap='jet')







