
#Importing the Important Modules
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("Machine Learning Application Project")
st.subheader("By - Rocco Peri")

st.markdown("This application explores the fundamental processes behind machine learning using three different models: linear regression, logistic regression, and decision trees.")
st.markdown("""In this application, you can...

**Model Selection**: Choose between three different machine learning models depending on your dataset
            
**Hyperparameter Tuning**: Choose different parameters for the decision tree model to see the impact the parameters have on model performance
            
**View Performance Metrics**: See model performance metrics and displays such as accuracy, R^2 score, precision, recall, F1score, etc.
You are free to use whatever datasets you like in this machine learning application, however...
            
1. Make sure that the data you upload is already preprocessed (no NaN values, no string values in a linear regression, binary variable as the target variable for 
logistic regression and/or decision tree). There are two preprocessed datasets on my **[Github](https://github.com/roccoperi/PERI-Data-Science-Portfolio/tree/main/MLStreamlitApp)** in the "sample data" folder
2. Ensure that you choose the appropriate model for your dataset. 
For example, for linear regression, your target variable can be any number, while logistic regression and decision trees require a binary target variable""")

st.markdown("""For the sample datasets, there are two examples through which you can test the models below. the "cleaned_titanic.csv" file contains the **[Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)**
without all of the missing values such that the regression can take place. I recommend using the titanic dataset for testing logistic regression and decision tree machine learning models, as the target variable of "survival" 
            is binary and thus can be classified by the other feature variables. The variable key is down below. 

- pclass: Ticket Class of Passengers: 1 = 1st, 2 = 2nd, 3 = 3rd

- age: Age in Years of the Passengers

- sibsp: # of Siblings/Spouses aboard the Titanic

- parch: # of parents/children aboard the Titanic

- fare: Passenger Fare

- survival: Whether the Passenger Survived or Not: 1 = yes, 0 = no
            
 The other dataset in the folder is the iris-numeric-dataset.csv, which is a dataset containing features about petals and sepals and how those aspects relate to the breed of flower. You can find the documentation **[here](https://www.kaggle.com/datasets/niranjandasmm/irisnumericdatasetcsv)**.
 I would recommend using this dataset for the linear regression model due to the non-binary target variable which is the class of flower the plant belongs to.""")

            
#Uploading Dataset
st.markdown("To start, please upload your own csv dataset")
uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    #Select the X and Y Variable to be used in the regression
    st.markdown("Please select the Feature (X) and Target (Y) Variables for the machine learning model")
    x_vars = st.multiselect("Select one or more X variables", numeric_columns)
    y_var = st.selectbox("Select the Y variable", numeric_columns)

    X = df[x_vars]
    Y = df[y_var]

    st.write("Feature variable preview:")
    st.write(X.head())

    st.write("\nTarget variable preview:")
    st.write(Y.head())

    #Model Selection
    st.markdown("What model would you like to choose to test the effect of X on Y?")
    model_type = st.radio("Data Type", options=["Linear Regression", "Logistic Regression", "Decision Tree"])
    # Choosing Linear Regression
    if model_type == "Linear Regression":
        data_type = st.radio("Data Type", options=["Unscaled", "Scaled"])
        #Choosing Unscaled Data
        if data_type == "Unscaled": 
            #Splitting the Data Into Test and Train Data
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            lin_reg_unscaled = LinearRegression()
            lin_reg_unscaled.fit(X_train_raw, y_train)
            y_pred_unscaled = lin_reg_unscaled.predict(X_test_raw)
            #Calculating and Displaying the Summary Statistics
            mse_raw = mean_squared_error(y_test, y_pred_unscaled)
            rmse_raw = root_mean_squared_error(y_test, y_pred_unscaled)
            r2_raw = r2_score(y_test, y_pred_unscaled)
            st.write("Unscaled Data Model:")
            st.write(f"Mean Squared Error: {mse_raw:.2f}")
            st.write(f"Root Squared Error: {rmse_raw:.2f}")
            st.write(f"R² Score: {r2_raw:.2f}")
            st.write("Model Coefficients (Unscaled):")
            st.write(pd.Series(lin_reg_unscaled.coef_, index=X.columns))
            st.write("Coefficient Interpretation")
            st.markdown("""
            Positive Coefficient: a one unit increase in the X variable leads to an increase in the Y variable proportional to the coefficient
                        
            Negative Coefficient: a one unit increase in the X variable leads to a decrease in the Y variable proportional to the coefficient""")
        #Choosing Scaled Data
        st.write("The purpose of rescaling the data is to put all the feature variables on the same scale with the same mean and standard deviation, leading to more accurate coefficient estimates.")
        if data_type == "Scaled": 
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
            lin_reg_scaled = LinearRegression()
            lin_reg_scaled.fit(X_train_scaled, y_train_scaled)
            y_pred_scaled = lin_reg_scaled.predict(X_test_scaled)
            mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled)
            r2_scaled = r2_score(y_test_scaled, y_pred_scaled)
            rmse_scaled = root_mean_squared_error(y_test_scaled, y_pred_scaled)
            st.write("Scaled Data Model:")
            st.write(f"Mean Squared Error: {mse_scaled:.2f}")
            st.write(f"Root Squared Error: {rmse_scaled:.2f}")
            st.write(f"R² Score: {r2_scaled:.2f}")
            st.write("Key Performance Metrics")
            st.markdown("""
            R²: Proportion of Variance in Y explained by X variables
            Root Mean Squared Error: How far predictions deviate from actual target, on average
            Mean Squared Error: Average squared difference between the residuals of the model""")
            st.write("Model Coefficients (Scaled):")
            st.write(pd.Series(lin_reg_scaled.coef_, index=X.columns))
    if model_type == "Logistic Regression":
        #Splitting the Data Into Test and Train Data
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        log_reg = LogisticRegression()
        log_reg.fit(X_train_raw, y_train)
        y_pred_log = log_reg.predict(X_test_raw)
        #Evaluating the Model
        y_pred = log_reg.predict(X_test_raw).round()
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")
        # Generate confusion matrix
        cm1 = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ax = sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)
        # Display classification report
        st.dataframe(classification_report(y_test, y_pred, output_dict = True))
        coef = pd.Series(log_reg.coef_[0], index=X.select_dtypes(include=['number']).columns.tolist())
        intercept = log_reg.intercept_[0]
        st.write("Key Performance Metrics")
        st.markdown("""
        - Accuracy: Overall percentage of correct classifications
        - Precision: Positive predictive value 
            - Of all predicted values, how many were actually positive?
        - Recall: True Positive Rate 
            - Of all actual positives, how many did the model correctly identify?
        - F1-Score: Balanced Mean of Precision and Recall""")

        # Display coefficients
        st.write("Model Coefficients:")
        st.write(coef)
        st.write("\nIntercept:", intercept)
        st.write("Coefficient Interpretation")
        st.markdown("""
        Positive Coefficient: increase in the log-odds (probability) of the target variable being 1
                    
        Negative Coefficient: decrease in the log-odds (probability) of the target variable being 1""")

    if model_type == "Decision Tree": 
        tree_type = st.radio("Type of Decision Tree", options=["Preset Hyperparameters", "Customized Hyperparameters", "Best Combination of Hyper Parameters"])
        if tree_type == "Preset Hyperparameters": #gini, best, no max depth, min samples split = 2, min samples leaf  = 1
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            desc_tree = DecisionTreeClassifier() # change default parameters
            desc_tree.fit(X_train, y_train)
            y_pred_desc = desc_tree.predict(X_test).round()
            # Evaluating the Model
            accuracy = accuracy_score(y_test, y_pred_desc)
            st.write(f"Accuracy: {accuracy:.2f}")
            # Confusion Matrix
            cm2 = confusion_matrix(y_test, y_pred_desc)
            fig, ax = plt.subplots()
            ax = sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)
            # Display classification report
            st.dataframe(classification_report(y_test, y_pred_desc, output_dict = True))
            st.write("Key Performance Metrics")
            st.markdown("""
            - Accuracy: Overall percentage of correct classifications
            - Precision: Positive predictive value 
                - Of all predicted values, how many were actually positive?
            - Recall: True Positive Rate 
                - Of all actual positives, how many did the model correctly identify?
            - F1-Score: Balanced Mean of Precision and Recall""")
        if tree_type == "Customized Hyperparameters":
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            s_criterion = st.selectbox("Select the Information Criterion", ("gini", "entropy", "log_loss"))
            s_max_depth = st.slider("Select maximum depth of the decision tree", min_value=1, max_value=20, step=1, value=1)
            s_minimum_samples_split= st.slider("Select the minimum amount of samples to be in a node to split", min_value=2, max_value=30, step=1, value=2)
            s_minimum_samples_leaf= st.slider("Select the minimum amount of samples to be in a leaf node", min_value=1, max_value=30, step=1, value=1)
            desc_tree = DecisionTreeClassifier(criterion=s_criterion, max_depth = s_max_depth, min_samples_split=s_minimum_samples_split, min_samples_leaf=s_minimum_samples_leaf)
            desc_tree.fit(X_train, y_train)
            y_pred_desc = desc_tree.predict(X_test).round()
            # Evaluating the Model
            accuracy = accuracy_score(y_test, y_pred_desc)
            st.write(f"Accuracy: {accuracy:.2f}")
            # Confusion Matrix
            cm2 = confusion_matrix(y_test, y_pred_desc)
            fig, ax = plt.subplots()
            ax = sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)
            # Display classification report
            st.dataframe(classification_report(y_test, y_pred_desc, output_dict = True))
            st.write("Key Performance Metrics")
            st.markdown("""
            - Accuracy: Overall percentage of correct classifications
            - Precision: Positive predictive value 
                - Of all predicted values, how many were actually positive?
            - Recall: True Positive Rate 
                - Of all actual positives, how many did the model correctly identify?
            - F1-Score: Balanced Mean of Precision and Recall""")
        if tree_type == "Best Combination of Hyperparameters":
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            param_grid = {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_depth': [None, 2, 3, 4, 5, 6],
                'min_samples_split': [2, 4, 6, 8],
                'min_samples_leaf': [1, 2, 3, 4]}
            dtree = DecisionTreeClassifier(random_state=42)
            grid_search = GridSearchCV(estimator = dtree,
                                    param_grid = param_grid,
                                    cv = 5,
                                    scoring='f1')
            grid_search.fit(X_train, y_train)
            st.write("Best parameters:", grid_search.best_params_)
            st.write("Best cross-validation score:", grid_search.best_score_)
            # Get the best estimator
            best_dtree = grid_search.best_estimator_
            y_pred_best = best_dtree.predict(X_test)
            st.write("Confusion Matrix:")
            cm3 = confusion_matrix(y_test, y_pred_best)
            fig, ax = plt.subplots()
            ax = sns.heatmap(cm3, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)
            st.write("Classification Report:")
            st.dataframe(classification_report(y_test, y_pred_best, output_dict = True))
            st.write("Key Performance Metrics")
            st.markdown("""
            - Accuracy: Overall percentage of correct classifications
            - Precision: Positive predictive value 
                - Of all predicted values, how many were actually positive?
            - Recall: True Positive Rate 
                - Of all actual positives, how many did the model correctly identify?
            - F1-Score: Balanced Mean of Precision and Recall""")
            