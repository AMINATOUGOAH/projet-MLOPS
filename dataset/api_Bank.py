import joblib
import uvicorn
from fastapi import FastAPI
import pandas as pd
from prometheus_client import Counter, make_asgi_app

exited_counter = Counter(
    'exited_counter', 'Counter for exited')
not_exited_counter = Counter(
    'not_exited_counter', 'Counter for not exited')

app = FastAPI()
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/Bank_Customer_Churn_Prediction")
def prediction_api(
        CreditScore: float, Age: float, Tenure: float,
        Balance: float, NumOfProducts: float, EstimatedSalary: float,
        BalanceSalaryRatio: float, TenureByAge: float,
        CreditScoreGivenAge: float, HasCrCard: float,
        IsActiveMember: int, Geography_Spain: int,
        Geography_France: int, Geography_Germany: int,
        Gender_Female: int, Gender_Male: int):

    Bank_Customer_Churn_Prediction_model = joblib.load(
        "./Bank_Churn_Prediction_model.joblib")

    x = [
        CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary,
        BalanceSalaryRatio, TenureByAge, CreditScoreGivenAge, HasCrCard,
        IsActiveMember, Geography_Spain, Geography_France, Geography_Germany,
        Gender_Female, Gender_Male]
    prediction = Bank_Customer_Churn_Prediction_model.predict(
        pd.DataFrame(x).transpose())
    exited = int(prediction) == 1
    if exited:
        exited_counter.inc()
    else:
        not_exited_counter.inc()
    return exited


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000,)
