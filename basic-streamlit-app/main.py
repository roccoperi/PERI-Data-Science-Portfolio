#Importing relevant packages/software
import streamlit as st
import pandas as pd

#Title
st.title("Penguin Summary Statistic Calculator")

#Describing the app
st.write("### App Description")
st.write("This app allows the user to filter the penguins in the Palmer Penguins Dataset by species, and then the app calculates the average bill length, the average bill depth, and the average weight of the species.")

st.write("### Palmer Penguins Dataset")
#Importing and loading the Palmer Penguins Dataset
penguin_df = pd.read_csv("data/penguins.csv")
st.dataframe(penguin_df)

#Selecting a penguin type 
species = st.selectbox("Select a penguin species", penguin_df["species"].dropna().unique())
filtered_penguin = penguin_df[penguin_df["species"] == species]

#Calculating average bill length, average bill depth, and average body mass in grams for the species
avg_bill_length = filtered_penguin['bill_length_mm'].mean()
avg_bill_depth = filtered_penguin['bill_depth_mm'].mean()
avg_body_mass = filtered_penguin['body_mass_g'].mean()

#Displaying the results for the user
st.write(f"### Average Measurements for {species}:")
st.write(f"- **Average Bill Length:** {avg_bill_length:.2f} mm")
st.write(f"- **Average Bill Depth:** {avg_bill_depth:.2f} mm")
st.write(f"- **Average Body Mass:** {avg_body_mass:.2f} g")