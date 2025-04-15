# Machine Learning Application Project
Hello! In this project, I implemented and presented three machine-learning models the user can work with in my Streamlit app. Within these models, the user can hyper-tune the models such that they can see how tuning certain aspects of the ML model affects its overall performance. The user has the option of uploading their data that they can run the ML models on, but I have also provided sample introductory datasets that can allow the user to initially work with the data with relative ease. 

## Project Overview
This project aims to provide the user with a brief introduction to three extremely useful machine learning models: linear regression, logistic regression, and decision trees. All over the world, machine learning is becoming not only more mainstream but also more powerful in its ability to perform tasks and predict future outcomes. Examples of this include Meta AI, Chat GPT 4-o, Google Gemini, Palantir applications, government defence operations, and so on. Therefore, providing a brief introduction to machine learning models through my Streamlit app provides a glimpse into how ML models work, their application in predicting target variables, and how certain parameters can be fine-tuned for optimization purposes. In my app, the user is prompted to upload a CSV file from which the user picks the predictor variables (X) and the target variable (Y) that is going to be inputted into the machine learning model. Then, the user has the option to choose from three different ML models depending on the nature of their data and the values it contains. Finally, within the models, there are opportunities for hyper-tuning, using unscaled or scaled data, and the opportunity to see how the models performed in terms of key performance metrics and the confusion matrices. I hope that whoever views my Streamlit app or runs the program locally will obtain a basic knowledge of how ML models work and become inspired by the future applications of ML models. 

## Instructions
You can view my project in two ways: the Streamlit Cloud or running the Streamlit app locally. 
1. Streamlit Cloud: To access my Streamlit app via the Streamlit Cloud, use this **[link](https://peri-data-science-portfolio-bclqv6kiypneixue5ptnwv.streamlit.app/)**
2. Local Access: To access my Streamlit app locally, download the main.py file in this repository and open it in Visual Studio Code. Make sure that the file is in the directory you are currently working within VS code. If you have not already, install Streamlit on VS Code using the command *pip install streamlit*. Then, you will run the command *streamlit run .\main.py* to run my app on a local server. 

## App Features 
Models 
1. Linear Regression: The user can choose between unscaled or scaled data depending on whether they want more accurate coefficient estimates.

2. Logistic Regression

3. Decision Trees: The user has three options to customize the decision trees...
   
   1. Preset Hyperparameters: The decision tree with preset hyperparameters which includes using the Gini Impurity Criterion, no max depth, a minimum sample split of 2 and a minimum sample leaf size of 1.
   
   2. Customized Hyperparameters: The user has the option to adjust the aforementioned hyperparameters and see how a change in one or two of the parameters affects model performance.
   
   3. The Best Combination of Hyperparameters: The model is put through GridSearchCV to find the optimal hyperparameters to produce the best ML model.
   
## References 
- [Pandas Cheat](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- [Flower Dataset](https://www.kaggle.com/datasets/niranjandasmm/irisnumericdatasetcsv)

## Images
<img align="left" width="360" height="500" src="https://github.com/roccoperi/PERI-Data-Science-Portfolio/blob/main/MLStreamlitApp/photos/Selecting%20of%20Variables%20for%20ML%20Model.png"> 
<img align="left" width="360" height="500" src="https://github.com/roccoperi/PERI-Data-Science-Portfolio/blob/main/MLStreamlitApp/photos/Customized%20Decision%20Tree.png"> 

**From Left to Right:**

Selection of Variables       

Customizable Decision Tree 


