import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

superkart_api = Flask("SuperKart Sales Forecast API")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "superkart_model.joblib")
model = joblib.load(MODEL_PATH)

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

@superkart_api.get("/")
def home():
    return jsonify({"message": "SuperKart Sales Forecast API is running."})

@superkart_api.post("/v1/predict")
def predict_sales():
    property_data = request.get_json(silent=True)
    if not property_data:
        return jsonify({"error": "Request body must contain JSON data."}), 400

    missing = [col for col in FEATURE_COLUMNS if col not in property_data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    input_data = pd.DataFrame([{col: property_data[col] for col in FEATURE_COLUMNS}])
    prediction = float(model.predict(input_data)[0])
    return jsonify({"Predicted Sales Revenue": round(prediction, 2)})

@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    if "file" not in request.files:
        return jsonify({"error": "CSV file must be supplied using the 'file' field."}), 400

    input_data = pd.read_csv(request.files["file"])
    missing = [col for col in FEATURE_COLUMNS if col not in input_data.columns]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    predictions = model.predict(input_data[FEATURE_COLUMNS])
    prediction_dict = {
        str(index): round(float(prediction), 2)
        for index, prediction in zip(input_data.index, predictions)
    }
    return jsonify(prediction_dict)

if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
