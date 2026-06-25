import numpy as np
import pandas as pd
np.random.seed(42)

n = 3000

data = {


    "CustomerID":
    range(1001,1001+n),


    "Age":
    np.random.randint(18,70,n),



    "Gender":
    np.random.choice(
        ["Male","Female"],
        n
    ),



    "SubscriptionType":
    np.random.choice(
        ["Basic","Premium","Enterprise"],
        n,
        p=[0.5,0.35,0.15]
    ),



    "MonthlyCharges":
    np.random.randint(
        200,5000,n
    ),



    "TenureMonths":
    np.random.randint(
        1,72,n
    ),



    "SupportTickets":
    np.random.randint(
        0,15,n
    ),



    "PaymentMethod":
    np.random.choice(
        [
            "Credit Card",
            "UPI",
            "Net Banking",
            "Debit Card"
        ],
        n
    ),



    "SatisfactionScore":
    np.random.randint(
        1,11,n
    ),



    "UsageHoursPerWeek":
    np.random.randint(
        1,50,n
    )

}



df = pd.DataFrame(data)
df.to_csv("data.csv", index=False) 