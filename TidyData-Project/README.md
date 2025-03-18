# Tidy Data Project
Hello! In this project, I cleaned, implemented, and visualized the adjusted 2008 Olympic Medalists dataset to draw conclusions about the nature of the Olympics and what types of events are the most prioritized. If you want a more detailed version of the steps I took and all of the code involved in drawing my conclusion, there is a Jupyter Notebook above that details both my code and my commentary on such. 

## Project Overview
The goal of this project is to utilize Pandas functions and tidy principles to manipulate the data such that I can properly engage with what the data has to offer. When navigating the data science field, there will be an innumerable amount of times that the current dataset is not in the right form, hard to understand, or missing large chunks of data. This project deals with this issue from both angles; discovering the issues with the data and using the right solutions/methods to solve such issues. By the end of my data cleaning process, I achieved a csv file that, as outlined in the [Tidy Data Principles](https://vita.had.co.nz/papers/tidy-data.pdf), has "each variable is a column, each observation is a row, and each type of observational unit is a table." From this cleaned data, I created a pivot table that allowed me to create a bar chart and a histogram that provided some insight into what I was primarily interested in above. I learned a lot with this project, and I hope that I can translate what I learned for future projects. 

## Instructions
1. If you want to see my Jupyter Notebook, download the .ipynb file and the olympics_06_medalists.csv file in the *data* folder above. When you run the notebook, make sure that the .csv file is in the correct working directory when you open it in whatever coding software you prefer, or else the notebook will not import the csv. file properly. Also, you need to ensure that the proper dependencies are loaded in prior to running the rest of the notebook. These packages include "pandas" for all of the data handling, and "matplotlib.pyplot" and "seaborn" for all of the visualizations. 
2. If you want to look at the the cleaned csv file, the files "cleaned_olympic.csv" and "olympic_pivot.csv" can be found in the *data* folder above. 
3. If you want to look at the visualizations produced from my code, they will be both at the bottom of this README.me and in the images folder above. 

## Dataset Description
The dataset used was the [2008 Olympic Medalists](https://edjnet.github.io/OlympicsGoNUTS/2008/) dataset. Although the Github linkes is about where the medalists' birth places are and the amount of medals produced from those birth places, the data adapted from this Github for the sake of this data science project contained the athletes, all of the events in the 2008 Olympics, and the type of medal the athelete received at the Olympics. 

## References 
- [Pandas Cheat](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Tidy Data Principles](https://vita.had.co.nz/papers/tidy-data.pdf)

## Images
<img align="left" width="500" height="300" src="https://github.com/roccoperi/PERI-Data-Science-Portfolio/blob/main/TidyData-Project/images/Cleaning%20Code.png"> 

  *In Order*
    **Code Used to Clean Data**

<img align="left" width="500" height="300" src="https://github.com/roccoperi/PERI-Data-Science-Portfolio/blob/main/TidyData-Project/images/barchart.png"> 

    **Ascention of Total Medals by Event**

<img align="left" width="500" height="300" src="https://github.com/roccoperi/PERI-Data-Science-Portfolio/blob/main/TidyData-Project/images/histogram.png"> 

    **Histogram of Events by Total Medals**


