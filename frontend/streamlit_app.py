import os
import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860")
FEATURE_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒")
st.title("SuperKart Sales Forecast")
st.write("Predict product-store sales revenue using the deployed machine learning model.")

st.subheader("Online Prediction")
product_weight = st.number_input("Product Weight", min_value=0.0, value=12.0)
product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, value=0.05, format="%.3f")
product_mrp = st.number_input("Product MRP", min_value=0.0, value=150.0)
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])
product_id_char = st.selectbox("Product ID Prefix", ["FD", "DR", "NC", "Others"])
store_age_years = st.number_input("Store Age (Years)", min_value=0, value=15)
product_type_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

single_input = pd.DataFrame([{
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar_content,
    "Product_Allocated_Area": product_allocated_area,
    "Product_MRP": product_mrp,
    "Store_Size": store_size,
    "Store_Location_City_Type": city_type,
    "Store_Type": store_type,
    "Product_Id_char": product_id_char,
    "Store_Age_Years": store_age_years,
    "Product_Type_Category": product_type_category,
}])[FEATURE_COLUMNS]

if st.button("Predict Sales", type="primary"):
    try:
        response = requests.post(
            f"{BACKEND_URL}/v1/predict",
            json=single_input.to_dict(orient="records")[0],
            timeout=30,
        )
        if response.ok:
            prediction = response.json()["Predicted Sales Revenue"]
            st.success(f"Predicted Sales Revenue: {prediction:,.2f}")
        else:
            st.error(response.text)
    except requests.RequestException as exc:
        st.error(f"Unable to connect to the prediction API: {exc}")

st.subheader("Batch Prediction")
st_text = "Upload a CSV containing the same feature columns expected by the model."
st.write(st_text)

uploaded_file = st.file_uploader("Upload batch CSV", type=["csv"])
if uploaded_file is not None:
    batch_preview = pd.read_csv(uploaded_file)
    st.dataframe(batch_preview.head())

    if st.button("Predict Batch", type="primary"):
        uploaded_file.seek(0)
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={"file": (uploaded_file.name, uploaded_file, "text/csv")},
                timeout=60,
            )
            if response.ok:
                predictions = response.json()
                result = batch_preview.copy()
                result["Predicted_Sales_Revenue"] = [
                    predictions[str(index)] for index in result.index
                ]
                st.success("Batch prediction completed successfully.")
                st.dataframe(result)
                st.download_button(
                    "Download Predictions",
                    result.to_csv(index=False).encode("utf-8"),
                    "SuperKart_Batch_Predictions.csv",
                    "text/csv",
                )
            else:
                st.error(response.text)
        except requests.RequestException as exc:
            st.error(f"Unable to connect to the prediction API: {exc}")
