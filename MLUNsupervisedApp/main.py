#Importing the Important/Necessary Modules
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score

st.title("Unsupervised Machine Learning Application Project")
st.subheader("By - Rocco Peri")

st.markdown("This application explores the fundamental processes behind machine learning using three different models: Principal Components, K-Means Clustering, and Hierarchical Clustering.")
st.markdown("""In this application, you can...

**Model Selection**: Choose between three different unsupervised machine learning models. 
            
**Hyperparameter Tuning**: Choose different parameters for the models (n_clusters, k, linkage) to see the impact the parameters have on the model performance + presentation.
                      
**View Metrics and Displays**: See model metrics such as Explained Variance Ratio, Cumulative Explained Variance, Optimal K, Principal Component Projections, & Dendrograms. 
            
You are free to use whatever datasets you like in this machine learning application. In a previous machine learning project, I ultimately forced the users of my app to preprocess thier 
datasets prior to using my app, and my app would not have functioned properly without such preprocessing. However, in this new and improved version, I have added the means through which missing values can be taken out of the dataset and categorical variables can turn 
into numerical values such that the unsupervised machine learning models can interact with the data in a meaningful way. Thus, I hope that my app is a lot easier to navigate through, and 
I believe that you will learn a lot about unsupervised machine learning through my app. 
            
- I uploaded the non-preprocessed but simplified **[Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/)** dataset in my sample datasets folder along with the non-processed 
**[Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)**. Please feel free to explore my app using those two datasets, but you are free to utilize whatver dataset you would like to test 
the unsupervised machine learning models on.""")

#Uploading Dataset
st.markdown("To start, please upload your own csv dataset (or a sample dataset from my Github repository).")
uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) # Converting the csv into a Pandas Dataframe
    st.write("**Number of Missing Values by Column**")
    st.dataframe(df.isnull().sum()) # Creating a table showing the number of missing values in each column
    if df.isnull().sum().any() == 0: # Checking to see if there are any missing values in the dataset.
        st.write("Congratulations, you dataset does not have any missing values!")
    else:
        st.markdown("""Unfortunately, unsupervised machine learning algorithms cannot work properly with missing data. Therefore, I have provided different criteria below through which the missing values can be taken out of the dataset. 
                    Please do not try to calculate the mean or median of categorical variables. For example, if you are using the penguins.csv file, there will be an error if you attempt to impute the mean or median on the "species" column as 
                    the app cannot mathmatically take the mean of a collection of string variables.""")
        column = st.selectbox("Choose a column to fill", df.select_dtypes(include=['number', 'object']).columns) # User input to pick the columns that they would like to delete missing values from
        method = st.selectbox("How would you like to fill your missing data?", [
         "Original DF",  "Drop Rows", "Drop Columns (>50% Missing)", "Impute Mean", "Impute Median", "Impute Zero"], index = None, placeholder = "~")
        df_clean = df.copy()
        if method == "Original DF":
            pass  # Keep the data unchanged.
        elif method == "Drop Rows":
            # Remove all rows that contain any missing values.
            df_clean = df_clean.dropna()
        elif method == "Drop Columns (>50% Missing)":
            # Drop columns where more than 50% of the values are missing.
            df_clean = df_clean.drop(columns=df_clean.columns[df_clean.isnull().mean() > 0.5])
        elif method == "Impute Mean":
            # Replace missing values in the selected column with the column's mean.
            df_clean[column] = df_clean[column].fillna(df[column].mean())
        elif method == "Impute Median":
            # Replace missing values in the selected column with the column's median.
            df_clean[column] = df_clean[column].fillna(df[column].median())
        elif method == "Impute Zero":
            # Replace missing values in the selected column with zero.
            df_clean[column] = df_clean[column].fillna(0)
        df_clean # Showing the dataset as the user is manipulating it
        if df_clean.isna().values.any() == True: # Checking if there are any missing values left in the dataset. 
            st.write("Your dataset still appears to have missing values, please use the methods above to clean your dataset of them.")
        else: 
            st.write("Your dataset no longer has missing values, lets proceed with alternative data wrangling methods.")
            # Prompting the user to input whether they have categorical variables in the dataset
            categorical = st.selectbox("Do you have any categorical variables in your dataset?", ("Yes", "No"), index = None, placeholder = "~") 
            if categorical == "Yes": # If the user has categorical variables
                #Prompting the user to input thier categorical variables
                cat_vars = st.multiselect("Select the categorical variables you would like to convert to numeric:", options = df.columns.tolist()) 
                df_clean[cat_vars] = df_clean[cat_vars].astype('category') # Labeling the inputted variables as the type "category"
                df_clean[cat_vars] = df_clean[cat_vars].apply(lambda x: x.cat.codes) # Of those labelled variables, assigning a unique number to each unique string such that all the observations are a number
                df_clean # Showing the new dataset with converted categorical variables
                st.write("If you are finished converting the categorical variables into a numeric form, lets now proceed with our unsupervised learning models")

            if categorical == "No": # If the user does not have categorical variables
                st.write("You have no categorical variables to change. Lets proceed with our unsupervised learning models.")
           
            #Select the X and Y Variable to be used in the unsupervised learning model
            numeric_columns = df_clean.select_dtypes(include=['number', 'object']).columns.tolist() #Obtaining all the feature variables from the cleaned dataset

            st.markdown("Please select the Feature (X) and Target (Y) Variables for the Unsupervised Learning Model.")
            x_vars = st.multiselect("Select one or more X variables", numeric_columns) # Selecting the feature variables
            y_var = st.selectbox("Select the Y variable", numeric_columns) # Selecting the target variable

            X = df_clean[x_vars] # Getting all the feature variables into one dataframe
            feature_names = X.columns.tolist() # Obtaining the feature variable names
            Y = df_clean[y_var] # Getting the target variable in one dataframe
            target_names = pd.unique(df[y_var].dropna()) # Obtaining the target variable name, using the preprocessed df because we need the names prior to the categorical variable transformation
            

            st.write("Feature variable preview:")
            st.write(X.head()) 

            st.write("\nTarget variable preview:")
            st.write(Y.head())

            #Model Selection
            st.markdown("What unsupervised learning model would you like to choose to test the effect of X on Y?")
            model_type = st.radio("Data Type", options=["Principal Components", "K-Means Clustering", "Hierarchical Clustering"])

            if model_type == "Principal Components":
                st.markdown("""Principal Component Analysis (PCA) is the an unsupervised learning technique for dimensionality reduction, which is the process of simplifying complex datasets such 
                        that the model is able to keep the majority of the information/results from the highly complex model, but with a lower total dimension for the sake of computational efficiency and better model performance.
                        For example, if we are able to simplify a model with 25 feature model into 3-4 components with similar performances, then the model with the components may be prefered for especially large datasets that 
                        requires a lot of computational power to run machine learning models on.""")
            
                st.markdown("""PCA uses the concept of explained variance to measure model performance with components. Explained variance is the amount of variance in the data that each principal component explains. Below, you will be able 
                             to adjust the number of components and see how the cumulative variance increases but at a diminishing rate with each additional component.""")
               
                #Centering and scaling the Data to eliminate potential biases within our dataset
                scaler = StandardScaler()
                X_std = scaler.fit_transform(X)

                #Loading the Principal Component Analysis w/ 2 Components 
                st.markdown("""Note: The max number of components cannot be greater than the amount of features in the model.""")
                components = st.slider("Choose the number of components you would like to include in the model", min_value = 2, max_value = len(X.columns), step = 1, value = 2) # User input for the number of components
                pca = PCA(n_components=components) # Setting the number of components in the PCA to the user inputted slider
                X_pca = pca.fit_transform(X_std) # Fitting the PCA to the scaled data

                #Displaying the Explained Variance Ratio
                explained_variance = pca.explained_variance_ratio_ # Obtaining the explained variance per component in the model
                st.write("Explained Variance Ratio:", explained_variance)
                st.write("Culumative Explained Variance:", np.cumsum(explained_variance)) 

                st.markdown("""Below is the Scatterplot fo the PCA Scores, detailing how the PCA separates the datapoints based on thier scores under the principal components. For the two-dimension scatterplot, I had to 
                            reduce the number of components to two.""")
                
                #Scatterplot of the PCA Scores 
                pca_s = PCA(n_components=2) # Setting the number of components to 2 for the scatterplot
                X_pca_s = pca_s.fit_transform(X_std) # Fitting the scaled data to the new components

                #Plotting the PCA Scores
                fig, ax = plt.subplots(figsize=(8,6))
                colors = ['navy', 'darkorange']
                for color, i, target_name in zip(colors, [0, 1, 2], target_names):
                    ax.scatter(X_pca_s[Y == i, 0], X_pca_s[Y == i, 1], color=color, alpha=0.7,
                                label=target_name, edgecolor='k', s=60)
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.set_title('PCA: 2D Projection of Dataset')
                ax.legend(loc='best')
                st.pyplot(fig)

                st.markdown("""Another major visualization of the PCA is the Biplot, which shows which direction the features (loadings) had the greatest effect on the variance captured by the principal components. Below is a depiction of the Biplot.""")

                loadings = pca.components_.T  # Compute the loadings 
                scaling_factor = 50.0  # Increased scaling factor by 5 times

                #Plotting the Biplot
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                # Plot the PCA scores as before
                for color, i, target_name in zip(colors, [0, 1], target_names):
                    ax2.scatter(X_pca_s[Y == i, 0], X_pca_s[Y == i, 1], color=color, alpha=0.7,
                                label=target_name, edgecolor='k', s=60)
                # Plot the loadings as arrows
                for i, feature in enumerate(feature_names):
                    ax2.arrow(0, 0, scaling_factor * loadings[i, 0], scaling_factor * loadings[i, 1],
                            color='r', width=0.02, head_width=0.1)
                    ax2.text(scaling_factor * loadings[i, 0] * 1.1, scaling_factor * loadings[i, 1] * 1.1,  # Adjusted text position
                            feature, color='r', ha='center', va='center')
                ax2.set_xlabel('Principal Component 1')
                ax2.set_ylabel('Principal Component 2')
                ax2.set_title('Biplot: PCA Scores and Loadings')
                ax2.legend(loc='best')
                ax2.grid(True)
                st.pyplot(fig2)

                st.markdown("Another visualization that helps is the Scree plot, which details the amount of cumulative explained variance is gained per principal component added. You can see such below. You want to see the elbow, " \
                "which is the point where adding another principal component is not going to add much to the cumulative explained variance. At that point, it may be optimal to use the amount of principal components detailed in the graph for analysis""")
                #Scree Plot
                pca_full = PCA(n_components = len(X.columns)).fit(X_std) # Total number of components for the plot 
                cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_) #Calculating the cululative variance
                #Plotting the Scree Plot
                fig3, ax3 = plt.subplots(figsize=(8, 6))
                ax3.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
                ax3.set_xlabel('Number of Components')
                ax3.set_ylabel('Cumulative Explained Variance')
                ax3.set_title('PCA Variance Explained')
                ax3.set_xticks(range(1, len(cumulative_variance)+1))
                ax3.grid(True)
                st.pyplot(fig3)

                st.markdown("With the PCA, we can now use the reduced dataset to analyze whether simplifying the data leads to better model performance with a Logistic Regression Model. With the # of principal components you inputted above," \
                "a Logistic Regression Model will use both the original data and the reduced data from PCA to make predictions about the target variable. The accuracy of those regressions is below, and you can see how changing the # of principal components changes the model accuracy.")
                #Model Comparision to a Logistic Regression Model
                X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_std, Y, test_size=0.2, random_state=42) #Splitting the Data w/ Original Dataset

                X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, Y, test_size=0.2, random_state=42) #Splitting the Data w/ PCA Dataset

                #Logistic Regression Results w/ Original Data
                clf_orig = LogisticRegression()
                clf_orig.fit(X_train_orig, y_train)
                y_pred_orig = clf_orig.predict(X_test_orig)
                acc_orig = accuracy_score(y_test, y_pred_orig)
                st.write("Logistic Regression Accuracy on Original Data: {:.2f}%".format(acc_orig * 100))

                # Logistic Regression Results on PCA Data
                clf_pca = LogisticRegression()
                clf_pca.fit(X_train_pca, y_train)
                y_pred_pca = clf_pca.predict(X_test_pca)
                acc_pca = accuracy_score(y_test, y_pred_pca)
                st.write("Logistic Regression Accuracy on PCA Data: {:.2f}%".format(acc_pca * 100))

                if acc_pca > acc_orig: 
                    st.write("The Logistic Regression Model performs better of the PCA Reduced Data vs. the Original Data")
                else: 
                    st.write("The Logistic Regression Model did not perform better on the PCA Reduced Data vs. the Original Data")


            if model_type == "K-Means Clustering":
                st.markdown("""Another unsupervised learning model is K-means clustering, which is the process of using the features of the data to group data points to k-clusters. 
                            In doing so, we make meaningful subgroups out of the data, find patterns not apparent from the data at a first glance, and can even draw interpretations from the clustering based on the target variable.""")
                
                #Centering and scaling the Data to eliminate potential biases within our dataset
                scaler = StandardScaler() 
                X_std = scaler.fit_transform(X)

                st.markdown("""First, choose the amount of initial centroids that will serve as the initial guesses for the cluster centers.""")
                st.markdown("""Note: If you have 2 classes in your dataset (ex. male and female under "sex" category in the penguins.csv file), then you should set k = 2.""")
                k = st.slider("Choose the amount of initial centroids:", min_value = 2, max_value = 10, step = 1, value = 2) #User input for the amount of clusters for k-means clustering
                kmeans = KMeans(n_clusters = k, random_state = 42) #Initializing the amount of clusters with the algorithm
                clusters = kmeans.fit_predict(X_std) #Creating the clusters

                st.markdown("""Here are the centroids of the data.""")
                st.write("Centroids:\n", kmeans.cluster_centers_)

                st.markdown("""To visualize how the k-means partitioned the data, we are going to reduce our data to two dimensions (n = 2) and use Principal Component Analysis to visualize the clusters.""")

                pca = PCA(n_components=2) # Setting the number of components to 2
                X_pca = pca.fit_transform(X_std) # Doing PCA on the scaled data

                #2D Scatter Plot of Clustering Results Using PCA
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['navy', 'darkorange', 'black', 'red', 'tomato', 'sandybrown', 'aqua', 'crimson', 'chocolate', 'khaki']
                for i in range(k):
                    ax.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1], color=colors[i], alpha=0.7,
                                label= f'Cluster {i+1}', edgecolor='k', s=60)
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.set_title('KMeans Clustering: 2D PCA Projection')
                ax.legend(loc='best')
                ax.grid(True)
                st.pyplot(fig)

                st.markdown("""Hopefully through this plot, you are able to understand how K-means clustering groups together datapoints based on the initial centriods.""")

                st.markdown("""Now, we are going to add the true labels (k is based on the number of unique labels in the dataset) onto the same 2D projection to see whether the classes in the target variable correllate to the different clusters.""")
               
                # For comparison, visualize true labels using PCA (same 2D projection)
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                colors = ['navy', 'darkorange', 'black', 'red', 'tomato', 'sandybrown', 'aqua', 'crimson', 'chocolate', 'khaki']
                for i, target_name in enumerate(target_names):
                    ax2.scatter(X_pca[Y == i, 0], X_pca[Y == i, 1],
                                color=colors[i], alpha=0.7, edgecolor='k', s=60, label=target_name)
                ax2.set_xlabel('Principal Component 1')
                ax2.set_ylabel('Principal Component 2')
                ax2.set_title('True Labels: 2D PCA Projection')
                ax2.legend(loc='best')
                ax2.grid(True)
                st.pyplot(fig2)

                
                st.markdown("To understand how well our clusters match our true labels, we calculate the accuracy score of our clusters which takes the higher accuracy score of the original labels and its compliment.")

                #Accuracy Score
                kmeans_accuracy = accuracy_score(Y, clusters) # Calculating the Accuracy Score

                st.write("Accuracy Score: {:.2f}%".format(kmeans_accuracy * 100)) # Turning the score into a percentage.
                st.write("Your k-means clustering model matched {:.2f}% of the true labels on the target variable.".format(kmeans_accuracy * 100))

                st.markdown("""If you want to find the optimal k for your model, you can look at the plot the Silhouette Scores of the model. The model with the highest silhouette score is the optimal k given the feature variables.""")
                # Defining the range of k values to try
                ks = range(2, 11) #Starting from 2 clusters to 11 clusters

                sil_scores = []  #Silhouette scores for each k

                # Loop over the range of k values
                for k in ks:
                    km = KMeans(n_clusters=k, random_state=42) # Initializing the number of initial centroids
                    km.fit(X_std) # Fitting the # of centiords to our scaled data
                    labels = km.labels_ 
                    sil_scores.append(silhouette_score(X_std, labels))


                # Plot the Silhouette Method result
                fig3, ax3 = plt.subplots(figsize=(12, 5))
                ax3.plot(ks, sil_scores, marker='o')
                ax3.set_xlabel('Number of clusters (k)')
                ax3.set_ylabel('Silhouette Score')
                ax3.set_title('Silhouette Score for Optimal k')
                ax3.grid(True)
                st.pyplot(fig3)

                #Optimal K under the silhouette curve
                best_k = ks[np.argmax(sil_scores)]
                st.write(f"Best k by silhouette: {best_k}  (score={max(sil_scores):.3f})")

                st.markdown("""Based on the optimal **k** you found, you can now go back to the initial slider, change the amount of initial centriods, and see how the graph/accuracy changes.""")

            if model_type == "Hierarchical Clustering": 
                st.markdown("""Finally, the last unsupervised model in this app is Hierarchical Clustering, which builds a tree of nested clusers in a dendogram such that 
                            you can segment datasets with a variable k value and detect patterns or outliers for future analysis in the dataset.""")
                
                st.markdown("""Below you will find a dendogram of your dataset, which details the multi-level structure in the unlabeled data. If you see a lot of separate datapoints 
                            clustered together on the bottom levels of the tree, that should indicate the number of optimal k you should include in your model.""")
                
                scaler = StandardScaler() # Initializing the standard scaler to eliminate bias
                X_scaled = scaler.fit_transform(X) # Scaling the data to eliminate bias

                #Dendogram
                Z = linkage(X_scaled, method="ward")      # linkage matrix

                # Plotting the Dendogram
                fig, ax = plt.subplots(figsize=(20, 7))
                dendrogram(Z)
                ax.set_title("Hierarchical Clustering Dendrogram")
                ax.set_xlabel("Observations")
                ax.set_ylabel("Euclidian Distance")
                st.pyplot(fig)
            
                st.markdown("Now that we have the the dendogram above, please pick the amount of clusters you would like to include in your hierarchichal clustering model")
                k = st.slider("Select the amount of clusters you would like in your hierarchical clustering algorithm", min_value=2, max_value = 10, step = 1, value = 2)
                agg = AgglomerativeClustering(n_clusters=k, linkage="ward") # Initializing the Hierarchical Clustering Model
                X["Cluster"] = agg.fit_predict(X_scaled) # Applying the model to our scaled data
                st.write("\nCluster sizes:\n", X["Cluster"].value_counts()) # Size of Each Cluster
                cluster_labels = X["Cluster"].tolist()

                st.markdown("Now, we can use Principal Component Analysis to graph the agglomerative clusters.")
                # For 2d Scatterplot, we use n_components = 2
                pca = PCA(n_components=2) # Initializing the PCA with the amount of clusters = 2
                X_pca = pca.fit_transform(X_scaled) # Fitting the PCA to our scaled data

                #Plotting the Agglomerative Clustering via PCA
                fig2, ax2 = plt.subplots(figsize=(10, 7))
                scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=60, edgecolor='k', alpha=0.7)
                ax2.set_xlabel('Principal Component 1')
                ax2.set_ylabel('Principal Component 2')
                ax2.set_title('Agglomerative Clustering on Uploaded Data (via PCA)')
                ax2.legend(*scatter.legend_elements(), title="Clusters")
                ax2.grid(True)
                st.pyplot(fig2)

                st.markdown("""Additionally, we can find the optimal **k** via the Silhouette  Analysis that was covered in the K-means clustering segment. 
                            As I explained earlier, the highest silhouette score indicates the optimal k for the model.""")
                
                # Range of candidate cluster counts
                k_range = range(2, 11)     # try 2–10 clusters; adjust as you like
                sil_scores = []

                for k in k_range:
                    # Fit hierarchical clustering with Ward linkage (same as dendrogram)
                    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_scaled)

                    # Silhouette: +1 = dense & well‑separated, 0 = overlapping, −1 = wrong clustering
                    score = silhouette_score(X_scaled, labels)
                    sil_scores.append(score)

                # Plot the silhouette curve
                fig3, ax3 = plt.subplots(figsize=(7,4))
                ax3.plot(list(k_range), sil_scores, marker="o")
                ax3.set_xticks(list(k_range))
                ax3.set_xlabel("Number of Clusters (k)")
                ax3.set_ylabel("Average Silhouette Score")
                ax3.set_title("Silhouette Analysis for Agglomerative (Ward) Clustering")
                ax3.grid(True, alpha=0.3)
                st.pyplot(fig3)

                #Optimal K under the silhouette curve
                best_k = k_range[np.argmax(sil_scores)]
                st.write(f"Best k by silhouette: {best_k}  (score={max(sil_scores):.3f})")

                st.markdown("""Based on the optimal **k** you found, you can now go back to the initial slider, change the k value, and see how the graph changes.""")
