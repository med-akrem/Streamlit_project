import streamlit as st
from PIL import Image # Import Image from Pillow
import pandas as pd
import numpy as np

#Title
st.title("Hello Streamlit!")
#Header and Subheader:
st.header("Welcome to Your First Streamlit App") # Adds a section header
st.subheader("This is a subheader")
#Text
st.text("This is your first interactive data app using Python!") # Displays text
st.text_area('Description') #Description area
st.success('bonjour')
st.error("WRONG")
st.info("info")
st.warning("be carful")
img = Image.open("test.png") # Open the image file
st.image(img, width=200) # Display the image with a specified width
if st.checkbox("Show/Hide"):
# display the text if the checkbox returns True value
    st.text("Showing the widget")
# Create a radio button to select gender
status = st.radio("Select Gender: ", ('Male', 'Female'))
if (status == 'Male'):
    st.error("Male")
else:
    st.success("Female")

st.title("Selection Box:")
# Create a dropdown menu for selecting a hobby
hobby = st.selectbox("Hobbies: ", ['Dancing', 'Reading', 'Sports'])
# Display the selected hobby
st.write("Your hobby is: ", hobby)

hobbies= st.multiselect("Select Your Hobbies:", ['Dancing', 'Reading', 'Sports'])# Display the number of selected hobbies
st.write("You selected", len(hobbies), 'hobbies')

if(st.button("About")):
    st.text("Welcome To Gomycode!!!")


df=pd.DataFrame(data=np.random.randn(100,3),columns=["A","B","c"])
st.dataframe(df.head(5))
st.write(df)


#############################################

name =st.text_input("enter your name")
if (st.button("submit")):

    st.success(f"hello, {name}")
st.date_input("enter un birth ")
st.time_input("meeting time ")



# dataframe
a = st.sidebar.number_input("pick a number ",0)
df=pd.DataFrame(data=np.random.randn(100,3),columns=["A","B","c"])
st.dataframe(df.head(a))

evel = st.slider("Choose a level", min_value=1, max_value=5)
# Display the selected level
st.write(f"Selected level: {evel}")

st.dataframe(df.head(evel))

# Tunisia approximate coordinates (centered near Tunis)
tunisia_lat = 36.8 # latitude
tunisia_lon = 10.18 # longitude
# Create random data around Tunisia
map_data = pd.DataFrame(
  np.random.randn(1000, 2) / [50, 50] + [tunisia_lat, tunisia_lon],
  columns=['lat', 'lon']
)
# Display the map
st.map(map_data, zoom=6)

st.sidebar.title("bonjour")

evel1 = st.sidebar.slider("Choose a level", min_value=1, max_value=5)