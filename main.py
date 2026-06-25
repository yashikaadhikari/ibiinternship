# Customer Churn Prediction System

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)




np.random.seed(42)

n = 3000







churn_probability = (

    0.4*(df["SatisfactionScore"] < 5)

    +

    0.3*(df["TenureMonths"] < 12)

    +

    0.2*(df["SupportTickets"] > 8)

    +

    0.1*(df["UsageHoursPerWeek"] < 10)

)


df["Churn"] = np.where(

    np.random.random(n) < churn_probability,

    "Yes",

    "No"

)



print(df.head())


print("\nMissing Values")

print(df.isnull().sum())



df.drop_duplicates(
    inplace=True
)


encoder = LabelEncoder()


categorical = [

    "Gender",
    "SubscriptionType",
    "PaymentMethod",
    "Churn"

]


for col in categorical:

    df[col] = encoder.fit_transform(
        df[col]
    )



print("\nAfter preprocessing")

print(df.head())



print("\nChurn Count")

print(
    df["Churn"].value_counts()
)


plt.figure(figsize=(6,6))


df["Churn"].value_counts().plot(

    kind="pie",

    autopct="%1.1f%%"

)


plt.title(
    "Churn Distribution"
)

plt.ylabel("")

plt.show()




plt.figure(figsize=(7,5))


sns.countplot(

    data=df,

    x="SubscriptionType",

    hue="Churn"

)


plt.title(
    "Subscription Type vs Churn"
)


plt.show()


plt.figure(figsize=(10,6))


sns.heatmap(

    df.corr(),

    annot=True

)


plt.title(
    "Feature Correlation"
)


plt.show()



plt.figure(figsize=(7,5))


sns.histplot(

    df["MonthlyCharges"],

    bins=30

)


plt.title(
    "Monthly Charges Distribution"
)


plt.show()

X = df.drop(

    [
        "CustomerID",
        "Churn"
    ],

    axis=1

)



y = df["Churn"]




scaler = StandardScaler()


X_scaled = scaler.fit_transform(X)




X_train,X_test,y_train,y_test = train_test_split(

    X_scaled,

    y,

    test_size=0.2,

    random_state=42

)




model = RandomForestClassifier(

    n_estimators=100,

    random_state=42

)



model.fit(

    X_train,

    y_train

)




y_pred = model.predict(

    X_test

)




print("\nMODEL PERFORMANCE")

print("----------------")


print(

"Accuracy:",

accuracy_score(
    y_test,
    y_pred
)

)


print(

"Precision:",

precision_score(
    y_test,
    y_pred
)

)


print(

"Recall:",

recall_score(
    y_test,
    y_pred
)

)


print(

"F1 Score:",

f1_score(
    y_test,
    y_pred
)

)



print("\nConfusion Matrix")


print(

confusion_matrix(

    y_test,

    y_pred

)

)



print(

classification_report(

    y_test,

    y_pred

)

)





print("\nCUSTOMER CHURN PREDICTOR")




try:


    age = int(
        input("Age: ")
    )


    subscription = input(
        "Subscription (Basic/Premium/Enterprise): "
    )



    charges = float(

        input(
            "Monthly Charges: "
        )

    )


    tenure = int(

        input(
            "Tenure Months: "
        )

    )


    tickets = int(

        input(
            "Support Tickets: "
        )

    )


    satisfaction = int(

        input(
            "Satisfaction Score(1-10): "
        )

    )


    usage = int(

        input(
            "Usage Hours/week: "
        )

    )



    if satisfaction <1 or satisfaction>10:

        raise ValueError(
            "Invalid satisfaction score"
        )



    new_customer = pd.DataFrame({

        "Age":[age],

        "Gender":[0],

        "SubscriptionType":[

            {
            "Basic":0,
            "Premium":1,
            "Enterprise":2
            }.get(subscription,0)

        ],

        "MonthlyCharges":[charges],

        "TenureMonths":[tenure],

        "SupportTickets":[tickets],

        "PaymentMethod":[0],

        "SatisfactionScore":[satisfaction],

        "UsageHoursPerWeek":[usage]

    })




    new_customer = scaler.transform(

        new_customer

    )



    prediction = model.predict(

        new_customer

    )



    if prediction[0]==1:

        print(
            "\nCustomer Likely to CHURN"
        )

    else:

        print(
            "\nCustomer Likely to STAY"
        )



except ValueError as e:


    print(

        "Error:",e

    )


except Exception:


    print(

        "Invalid input. Try again."

    )