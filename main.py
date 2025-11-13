import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px 
import pickle

st.set_page_config(page_title="Diabet Protiction App " ,page_icon="💉")

#page title 
st.title("welcome to my web app ")
#name=st.text_input("enter you name")
#if name :
  #  st.success(f"Hi{name} welcome to my app ")

page = st.sidebar.radio("Go to", ["🏠 Home", "🧠 Model"])
if page == "🏠 Home":
  # ------------  KEEP DATA AFTER REFRESH ------------
    if "df" not in st.session_state:
        st.session_state.df = None
        #If there is NO variable named "df" saved in session_state, create it and set it to None.

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"]) 

    # if a new file is uploaded ⇒ read & save it
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
    # get the saved dataframe
    df = st.session_state.df



    # ------------  MAIN UI ------------
    if df is not None:
            # ✅ Clear Button here
        if st.button("Clear Data"):
            st.session_state.df = None
            st.rerun()

            
        st.success("File uploaded successfully!")

        if st.checkbox("show data"):
            st.dataframe(df.head()) # Display data
        #st.table(df) # Static table
        
        
            # Optional:we can add for data
        with st.expander("🔍 Data Preview"):
            st.dataframe(df.head())


            
        st.write("Shape:", df.shape) # prints this information to the app.
        st.write("Columns:", df.columns.tolist()) # lists all column names in the dataset

        #slider  
        no_rows =st.slider("select rows ", min_value=1, max_value=len(df))
        
        #multi-select box 
        choose_col = st.multiselect("Select columns to show :", df.columns.to_list(),default=df.columns.to_list())

        #display data     # Display selected rows + columns

        st.write(df[:no_rows][choose_col])

        

        # Expander 1 - Summary Statistics
        with st.expander("Summary Statistics"):
            st.write(df.describe())

        st.success("Statistical analysis complete!")


        # ---------------- VISUALIZATION ----------------

        st.subheader("Visualization")


        
        tab1, tab2 = st.tabs(["scatter plot ", "histogram"])
        with tab1:
            #creates a dropdown list.
            col1, col2 = st.columns(2)
            with col1:
             x_col = st.selectbox("Select X-axis", df.columns)
            with col2:
            #ensures the dropdown lists all columns in the dataset.
             y_col = st.selectbox("Select Y-axis", df.columns)

            #Creating a Scatter Plot
            fig_scatter = px.scatter(df, x=x_col, y=y_col, title="Scatter Plot")
            # Displaying the Plot
            st.plotly_chart(fig_scatter)
        with tab2:

            st.subheader("Histogram")

            hist_col = st.selectbox("Select column for histogram", df.columns)
            bins = st.slider("Number of bins", min_value=5, max_value=100, value=20)

            fig_hist = px.histogram(df,x=hist_col,nbins=bins, title=f"Histogram of {hist_col}")
            st.plotly_chart(fig_hist)
        
        
        
elif page == "🧠 Model":
 
    st.title("model  Page")
    st.title("🩺 Diabetes Prediction App")
    st.write("Enter the patient's medical details:")
    
    # ---------- INITIALISATION DU TABLEAU ----------
    if "patients_data" not in st.session_state:
        st.session_state.patients_data = []
        # Input fields for user data
    if "df" not in st.session_state:
        st.session_state.df = None
        #If there is NO variable named "df" saved in session_state, create it and set it to None.

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"]) 

    # if a new file is uploaded ⇒ read & save it
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
    # get the saved dataframe
    df = st.session_state.df

    #load the trained model
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
        
        
    
    
    # Input fields for user data
    col1,col2=st.columns(2)
    with col1:
        Preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=0,help="Number of times pregnant")
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=100,help="Blood sugar level (mg/dL)")
        bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70,help="Diastolic blood pressure (mm Hg)")
        skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20,help="Triceps skin fold thickness (mm)")
    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79,help="2-Hour serum insulin (mu U/ml)")
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0,help="Body mass index (weight in kg/(height in m)^2)")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5,help="Family history likelihood of diabetes")
        age = st.number_input("Age", min_value=1, max_value=120,    value=30,help="Age in years")
   
 
    #prepare the input data as a DataFrame
    input_data = pd.DataFrame({'Pregnancies': [Preg],
                              'Glucose': [glucose],
                              'BloodPressure': [bp],
                              'SkinThickness': [skin],
                               'Insulin': [insulin],
                               'BMI': [bmi],
                               'DiabetesPedigreeFunction': [dpf],
                               'Age': [age]})
    #display the input data
    st.subheader("Input Data")
    st.write(input_data)
    
    #add import features
    st.subheader("Important Features for Diabetes")
    importances = model.feature_importances_
    features=input_data.columns
    #crete a dataframe for feature importance
    feature_importance_df = pd.DataFrame ({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    with st.expander("👩🏻‍⚕️Important Features"):
        fig_importance = px.bar(
            feature_importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance',
            color='Importance',
            color_continuous_scale='Viridis')
        st.plotly_chart(fig_importance)
    
    
    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
    # ✅ Sauvegarder les valeurs saisies dans la session
        st.session_state.patients_data.append({
        "Pregnancies": Preg,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
        "Prediction": "Diabetic" if prediction[0] == 1 else "Non-Diabetic",
        "Probability": round(float(prediction_proba[0][1]), 3)
    })

        # Display results
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("⚠️ The model predicts that the patient **has diabetes**.")
        else:

            st.success("✅ The model predicts that the patient **does not have diabetes**.")

        # Show prediction probabilities
        st.progress(int(prediction_proba[0][1] * 100))
        st.subheader("Prediction Probabilities")
        st.write(f"Probability of No Diabetes: {prediction_proba[0][0]:.2%}")
        st.write(f"Probability of Diabetes: {prediction_proba[0][1]:.2%}")
         # risk level interpretation
        st.subheader("Risk Level Interpretation")
        if prediction_proba[0][1] < 0.2:
                st.success("Low Risk of Diabetes")
        elif 0.2 <= prediction_proba[0][1] < 0.5:
            st.warning("Moderate Risk of Diabetes")
        else:
            st.error("High Risk of Diabetes")
    # ---------- AFFICHER L’HISTORIQUE ----------
    if len(st.session_state.patients_data) > 0:
        st.subheader("🧾 Patients History")
        df_history = pd.DataFrame(st.session_state.patients_data)
        st.dataframe(df_history)

     # Bouton pour tout effacer
    if st.button("🗑️ Clear All History"):
        st.session_state.patients_data = []
        st.rerun()
    

        
    
    
                               
        