import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import MachineInput
from backend.model import model


app = FastAPI(
    title="Predictive Maintenance API",
    description="Machine failure prediction API",
    version="1.0.0"
)

 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "Predictive Maintenance Model",
        "version": "1.0.0",
        "target": "machine_failure"
    }


@app.post("/predict")
def predict_machine_failure(
    data: MachineInput
):

    temperature_difference = (
        data.process_temperature
        - data.air_temperature
    )

    power_proxy = (
        data.rotational_speed
        * data.torque
    )

    input_data = pd.DataFrame([
        {
            "machine_type": data.machine_type,
            "air_temperature": data.air_temperature,
            "process_temperature": data.process_temperature,
            "rotational_speed": data.rotational_speed,
            "torque": data.torque,
            "tool_wear": data.tool_wear,
            "temperature_difference": temperature_difference,
            "power_proxy": power_proxy
        }
    ])
    print(input_data)

    probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = int(
        probability >= 0.5
    )

    if prediction == 1:
        risk = "HIGH"
        message = (
            "Potential machine failure detected."
        )
    else:
        risk = "LOW"
        message = (
            "No machine failure detected."
        )

    return {
        # "prediction": prediction,
        # "failure": bool(prediction),
        # "probability": round(
        #     float(probability),
        #     4
        # ),
        # "risk": risk,
        # "message": message, 
        "input_data":input_data
    }
 