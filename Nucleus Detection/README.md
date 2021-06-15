# Nucleus Detection

## Spot Nuclei. Speed Cures.
Imagine speeding up research for almost every disease, from lung cancer and heart disease to rare disorders. Here I am creating an algorithm to automate nucleus detection.
We’ve all seen people suffer from diseases like cancer, heart disease, chronic obstructive pulmonary disease, Alzheimer’s, and diabetes. Many have seen their loved ones 
pass away. Think how many lives would be transformed if cures came faster.
By automating nucleus detection, you could help unlock cures faster—from rare disorders to the common cold.

## Why nuclei?
Identifying the cells’ nuclei is the starting point for most analyses because most of the human body’s 30 trillion cells contain a nucleus full of DNA, the genetic code
that programs each cell. Identifying nuclei allows researchers to identify each individual cell in a sample, and by measuring how cells react to various treatments, the
researcher can understand the underlying biological processes at work.

## Finding the Nucleus help to...
* Locate cells in varied conditions to enable faster cures
* Free biologist to focus on solutions
* Improve throughput for research and insight
* reduce time-to-market for new drugs- currently 10 years
* increase number of compounds for experiments
* improve health and increase quility of life


## Overview :-
Dectecting the Nucleus cells using U-NET model



## Data :-
[Kaggle Nuclei dataset](https://www.kaggle.com/c/data-science-bowl-2018/data)

## Run :-
* (In Google colab)
    * Go to your account, Scroll to API section and Click Expire API Token to remove previous tokens
    * Click on Create New API Token - It will download kaggle.json file on your machine.
    * Go to your Google Colab project file and run the following commands:
        * ```! pip install -q kaggle```
        * ```from google.colab import files```
            * files.upload()
            * Choose the kaggle.json file that you downloaded
        * ```! mkdir ~/.kaggle```
        * ```! cp kaggle.json ~/.kaggle/``` 

    * Make directory named kaggle and copy kaggle.json file there.
        * ```! chmod 600 ~/.kaggle/kaggle.json```

    * Change the permissions of the file.
        * ```! kaggle datasets list```
    * That's all ! You can check if everything's okay by running this command.

    * Download Data
         *  ``` ! kaggle competitions download -c 'name-of-competition' ```
         https://www.kaggle.com/c/data-science-bowl-2018/data

         * Use unzip command to unzip the data:\
            For example,
         * Create a directory named train,
            ```! mkdir train```
        * unzip train data there,
            ```! unzip train.zip -d train```


## Dependencies
* Tensorflow
* Scikit-image
* Numpy
* Matplotlib 

## Results :-
<p align="left">
<img src="https://github.com/Lalit78716/Image-segmentation-Projects/blob/main/Nucleus%20Detection/Screenshots/Screenshot%20(493).png"/>
<img src="https://github.com/Lalit78716/Image-segmentation-Projects/blob/main/Nucleus%20Detection/Screenshots/Screenshot%20(495).png",width="600" height="400"/>
</p>


## Accuracy :-
<p align="left">
<img src="https://github.com/Lalit78716/Image-segmentation-Projects/blob/main/Nucleus%20Detection/Screenshots/T_V_accuracy_rms%20(1).png"/>
<img src="https://github.com/Lalit78716/Image-segmentation-Projects/blob/main/Nucleus%20Detection/Screenshots/T_V_loass_rms%20(1).png"/>
</p>

