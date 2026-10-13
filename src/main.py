from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load trained model
model = joblib.load("./models/aml_model.pkl")


class Transaction(BaseModel):
    Amount: float
    Payment_currency: int
    Received_currency: int
    Sender_bank_location: int
    Receiver_bank_location: int
    Payment_type: int
    Hour: int
    Night_Flag: int
    Cross_Border: int
    Sender_account: int
    Receiver_account: int


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(transaction: Transaction):

    input_data = pd.DataFrame([{
        "Amount": transaction.Amount,
        "Payment_currency": transaction.Payment_currency,
        "Received_currency": transaction.Received_currency,
        "Sender_bank_location": transaction.Sender_bank_location,
        "Receiver_bank_location": transaction.Receiver_bank_location,
        "Payment_type": transaction.Payment_type,
        "Hour": transaction.Hour,
        "Night_Flag": transaction.Night_Flag,
        "Cross_Border": transaction.Cross_Border,
        "Sender_account": transaction.Sender_account,
        "Receiver_account": transaction.Receiver_account
    }])

    prediction = model.predict(input_data)

    return {
        "prediction": int(prediction[0])
    }