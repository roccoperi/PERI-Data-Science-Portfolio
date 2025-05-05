# Unsupervised Machine Learning Application Project
Hello! In my final project of my Introduction to Data Science Class, I uploaded a streamlit app to the Streamlit Community Cloud that allows the user to upload a dataset, handle missing values and categorical variables, and then input the data into three different unsupervised machine learning models: principal component analysis (PCA), k-means clustering, and hierarchical clustering. The app contains accuracy/performance statistics and graphical representations of the unsupervised machine learning models that can all be affected by the user tweaking the hyperparameters of the respective models. In the **[sample datasets](https://github.com/roccoperi/PERI-Data-Science-Portfolio/tree/main/MLUNsupervisedApp/sample%20datasets)** folder, you can find two introductory datasets that can be experimented with. However, the user can upload their dataset to interact with the three machine learning models.

## Project Overview
This project aims to show how unsupervised machine learning models can provide insight into unlabeled data. In a lot of professions, there are going to be times when you are given a dataset with either no labels or too many features, and thus the use of these unsupervised machine learning models is to either reduce the complexity/noise of the data through dimensionality reduction or cluster the points such that patterns and insights can be drawn from the seemingly incoherent dataset. Examples where unsupervised machine learning is most useful include social network analysis, customer segmentation, and fraud detection, but if a user has a certain goal, the widespread applicability of unsupervised machine learning models allows the use of such models to reach their goal. In my app, the user is prompted to upload a CSV file, and then the user handles the missing data in the dataset through 6 different methods of choice. Moreover, the user is then prompted to convert categorical variables of the dataset into numeric variables if they want to include those variables in the unsupervised learning models, and then ther user predicts the feature variables (X) and the target/diagnostic variable (Y) that is going to be inputted into the three unsupervised learning models. After selecting the feature and target variables, the user has the option to select one of the three models and see how each model interacts with and interprets the user-inputted data. Within these three unsupervised machine learning models, there exist opportunities for hyperturning (such as choosing the k-centroids to be included in the clustering models or selecting the # of components in the PCA model) and promoting user immersion and intellectual curiosity. I hope that the users of my Streamlit app with gain a basic understanding of unsupervised learning models and gain a sense of appreciation and wonder for how these models can be used in all aspects of life. 

## Instructions
You can view my project in two ways: the Streamlit Cloud or running the Streamlit app locally. 
1. Streamlit Cloud: To access my Streamlit app via the Streamlit Cloud, use this **[link](https://peri-data-science-portfolio-klr2q4kefbndjg4fcddwne.streamlit.app/)**
2. Local Access: To access my Streamlit app locally, download the main.py file in this repository and open it in Visual Studio Code. Make sure that the file is in the directory you are currently working in VS Code. If you have not already, install Streamlit on VS Code using the command *pip install streamlit*. Then, you will run the command *streamlit run .\main.py* to run my app on a local server. For Local Access, you need to install the proper dependencies. The dependencies and the versions that you need to install are located in the *[requirements.txt](https://github.com/roccoperi/PERI-Data-Science-Portfolio/blob/main/MLUNsupervisedApp/requirements.txt)*.

## App Features
### Handling Missing Data
The user has the option to choose any column in their uploaded dataset and fill the missing values in the column with different values based on 6 different choices. These choices include...
- Keep the Original Column
- Drop Rows with Missing Values
- Drop Columns with >50% of the Values Missing
- Impute the Mean of the Column for the Missing Values
- Impute the Median of the Column for the Missing Values
- Impute 0 for the Missing Values in the Column

### Handling Categorical Variables 
If the user has categorical variables that they would like to include in the unsupervised machine learning algorithm, then they have the option to select those variables to be converted into a numeric form 
 - Note: The assignment of such numeric values to the categories is done through labeling the columns as the type("category") and then using the pandas function *x.cat.codes*. For documentation, see **[here](https://pandas.pydata.org/docs/reference/api/pandas.Series.cat.codes.html).**

### Models 
1. Principal Component Analysis: The user can choose the number of components to be included in the PCA. 

2. K-means Clustering: The user can specify the number of k-centroids to be included in the model.

3. Hierarchical Clustering: The user can select the number of k-centroids to be included in the model.
   
## References 
- [Pandas Cheatsheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- [Penguins Dataset](https://allisonhorst.github.io/palmerpenguins/articles/intro.html)

## Images
**Dendrogram from Hierarchical Clustering**
<img align="center" width="900" height="500" src="https://github.com/roccoperi/PERI-Data-Science-Portfolio/blob/main/MLUNsupervisedApp/images/dendogram.png"> 

**Scree Plot from PCA**

<img align="center" width="800" height="600" src="https://github.com/roccoperi/PERI-Data-Science-Portfolio/blob/main/MLUNsupervisedApp/images/pca%20variance.png"> 


**True Labels Projection from K-Means Clustering**

<img align="center" width="800" height="600" src="https://github.com/roccoperi/PERI-Data-Science-Portfolio/blob/main/MLUNsupervisedApp/images/true_labels_kmeans.png"> 







