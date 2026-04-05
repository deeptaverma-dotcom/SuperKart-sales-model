import pandas as pd
import joblib
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="SuperKart Sales Prediction", layout="centered")

HF_MODEL_REPO = "DeeptaV/SuperKart-sales-model"
MODEL_FILE_NAME = "RandomForest_best_model.pkl"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=MODEL_FILE_NAME,
        repo_type="model"
    )
    model = joblib.load(model_path)
    return model

model = load_model()

st.title("SuperKart Sales Prediction")
st.write("Enter the product and store details below to predict total sales.")

product_id = st.text_input("Product Id", "FDX07")
product_weight = st.number_input("Product Weight", value=19.2)
product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_allocated_area = st.number_input("Product Allocated Area", value=0.065, format="%.6f")
product_type = st.text_input("Product Type", "Dairy")
product_mrp = st.number_input("Product MRP", value=249.8)
store_id = st.text_input("Store Id", "OUT049")
store_establishment_year = st.number_input("Store Establishment Year", value=1999, step=1)
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.text_input("Store Type", "Supermarket Type1")

if st.button("Predict Sales"):
    input_df = pd.DataFrame({
        "Product_Id": [product_id],
        "Product_Weight": [product_weight],
        "Product_Sugar_Content": [product_sugar_content],
        "Product_Allocated_Area": [product_allocated_area],
        "Product_Type": [product_type],
        "Product_MRP": [product_mrp],
        "Store_Id": [store_id],
        "Store_Establishment_Year": [store_establishment_year],
        "Store_Size": [store_size],
        "Store_Location_City_Type": [store_location_city_type],
        "Store_Type": [store_type],
        "__index_level_0__": [0]
    })

    prediction = model.predict(input_df)
    st.success(f"Predicted Product Store Sales Total: {prediction[0]:.2f}")
