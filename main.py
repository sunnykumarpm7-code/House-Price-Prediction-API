import io
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


app = FastAPI(
    title="California Housing Price Prediction API",
    description="Machine Learning API for predicting California housing prices",
    version="1.0.0"
)


# Load trained model and feature columns
model = joblib.load("random_forest_model.joblib")
features = joblib.load("model_columns.joblib")


# Input validation model
class HouseFeatures(BaseModel):

    MedInc: float = Field(
        gt=0,
        description="Median Income of the Neighborhood"
    )

    HouseAge: float = Field(
        gt=0,
        description="Median House Age in the Neighborhood"
    )

    AveRooms: float = Field(
        gt=0,
        description="Average Number of Rooms per Household"
    )

    AveBedrms: float = Field(
        gt=0,
        description="Average Number of Bedrooms per Household"
    )

    Population: float = Field(
        gt=0,
        description="Population of the Neighborhood"
    )

    AveOccup: float = Field(
        gt=0,
        description="Average Number of Occupants per Household"
    )

    Latitude: float = Field(
        ge=32,
        le=42,
        description="Latitude of the Neighborhood"
    )

    Longitude: float = Field(
        ge=-125,
        le=-114,
        description="Longitude of the Neighborhood"
    )


# Home endpoint
@app.get("/")
def home():

    return {
        "message": "Welcome to the California Housing Price Prediction API!",
        "status": "API is running successfully.",
        "endpoint": "Send a POST request to /predict"
    }


# Health check endpoint
@app.get("/health")
def health():

    return {
        "status": "API is running successfully.",
        "model": "Random Forest Regressor",
        "features": features,
        "average_absolute_error": "$32,773"
    }


# Single house prediction
@app.post("/predict")
def predict(house: HouseFeatures):

    try:

        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])

        # Ensure correct feature order
        input_data = input_data[features]

        predicted = model.predict(input_data)[0]

        # California Housing dataset target is in units of $100,000
        predicted_usd = predicted * 100000

        error_margin = 32773

        return {
            "predicted_price": f"${predicted_usd:,.2f}",

            "predicted_price_raw":
            f"{predicted:.2f} hundreds of thousands",

            "confidence_range":
            f"${predicted_usd - error_margin:,.2f} "
            f"to ${predicted_usd + error_margin:,.2f}"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# CSV file prediction
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):

    # Check file type
    if not file.filename.lower().endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a CSV file."
        )

    try:

        # Read uploaded CSV
        contents = await file.read()

        df = pd.read_csv(io.BytesIO(contents))


        required_columns = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude"
        ]


        # Check missing columns
        missing_columns = [

            col for col in required_columns

            if col not in df.columns

        ]


        if missing_columns:

            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing_columns}"
            )


        # Check empty file
        if df.empty:

            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV file is empty."
            )


        # Select required features
        input_data = df[required_columns]


        # Make predictions
        predictions = model.predict(input_data)


        # Convert predictions to USD
        df["Predicted_Price_USD"] = predictions * 100000


        # Format price
        df["Predicted_Price_USD"] = df[
            "Predicted_Price_USD"
        ].apply(
            lambda x: f"${x:,.2f}"
        )


        # Convert dataframe to CSV
        output = io.StringIO()

        df.to_csv(output, index=False)

        output.seek(0)


        # Return downloadable CSV
        return StreamingResponse(

            output,

            media_type="text/csv",

            headers={
                "Content-Disposition":
                "attachment; filename=predictions.csv"
            }

        )


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the file: {str(e)}"
        )