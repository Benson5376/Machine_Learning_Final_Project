# Machine Learning Final Project

## Reproducing my submission

### Environment Setup & Installation

Python version :```3.9.7```

All the packages that should be installed are in ```requirements.txt```

First download ```requirements.txt```

Build the virtual environment via   
```$ virtualenv -p <path to python version> myenv```  
```$ pip install -r requirements.txt```
## Download the code and csv files
Download the file below:  
```109550159_Final_train.py```  
```109550159_Final_inference.py```  
```sample_submission.csv```  
```test.csv```  
```train.csv```  
After downloading the files above, put them in the same folder.

## Training Code
You can modify the parameters and produced ```best_model.csv``` by running ```109550159_Final_train.py```  
![image](https://github.com/Benson5376/Machine-Learning-Final-Project/blob/main/iamge01.png)  

## Download the pre-trained model
Without running the training code, you can also download the pretrained model in the link below and put it in the smae folder as other files.  
Model link: https://drive.google.com/file/d/1ED7niJo8w2uVn-X9mkt8n_q09oskU0o8/view  
  
## Inference
After prdoducing the model by training code or downloading the model by the model link provided above, you can run ```109550159_Final_inference.py```
The program will produce submission.py, which is the final submission.
