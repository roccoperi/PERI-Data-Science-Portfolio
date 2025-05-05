# Basic Streamlit App 

Hello! In my first project for my Introduction to Data Science Class, I created a basic Streamlit app that previews the **[Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/articles/intro.html)** dataset, and the user has the option to select the species of penguin for analysis.  From the species the user selected, the app will output the average measurements for that penguin species, including average bill length, average bill depth, and average body mass.

## Project Overview
This project was my first step into everything that the Streamlit application has to offer. In this project, I only used basic functions like *st.write*, *st.selectbox*, and *st.title*, but looking back on this project after implementing two Streamlit apps with machine learning applications, I now know that there is so much within the Streamlit documentation that can provide for meaningful insight to the user. My app calculates the average summary statistics for a user-inputted specific species of penguin, and this project provided the foundation upon which I built my Streamlit skills and can now upload apps onto the Streamlit Cloud for other users to publicly enjoy. I am very happy I was exposed to Streamlit as a way to spread my love for Data Science to others.

## Instructions
You can view my project by running the Streamlit app locally. 
- Local Access: To access my Streamlit app locally, download the main.py and the penguins.csv (located in the "data" folder) and open the main.py file in Visual Studio Code. Ensure that the two files are present in the directory you are currently working in VS Code. If you have not already, install Streamlit on VS Code using the command *pip install streamlit*. Then, you will run the command *streamlit run .\main.py* to run my app on a local server. 

## App Features

### Data Preview 
- The user can view the penguins.csv file within my Streamlit App, detailing the aspects of the penguins that are recorded in the CSV file (ex., species, bill length, etc.)

### Average Summary Statistics 
- After the user selects the species of the penguin for analysis, the app outputs the average bill length, the average bill depth, and the average body mass of that species of penguin.

## References 
- [Pandas Cheatsheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Penguins Dataset](https://allisonhorst.github.io/palmerpenguins/articles/intro.html)

## Images
**Average Summary Statistics**
<img align="left" width="850" height="300" src="https://github.com/roccoperi/PERI-Data-Science-Portfolio/blob/main/basic-streamlit-app/images/output.png"> 
