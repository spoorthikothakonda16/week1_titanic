# ============================================================
# WEEK 1 - TITANIC SURVIVAL PREDICTION
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("titanic.csv")

print("=" * 60)
print("TITANIC DATASET")
print("=" * 60)

print("\nFirst 10 rows:")
print(df.head(10))


# ============================================================
# 2. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

df.info()


# ============================================================
# 3. BASIC STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("BASIC STATISTICS")
print("=" * 60)

print(df.describe())


# ============================================================
# 4. MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())


# ============================================================
# 5. HANDLE MISSING AGE
# ============================================================

if "Age" in df.columns:
    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)

    print("\nMissing Age values after median imputation:")
    print(df["Age"].isnull().sum())


# ============================================================
# 6. HANDLE MISSING EMBARKED
# ============================================================

if "Embarked" in df.columns:
    embarked_mode = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)

    print("\nMissing Embarked values after mode imputation:")
    print(df["Embarked"].isnull().sum())


# ============================================================
# 7. HANDLE MISSING FARE
# ============================================================

if "Fare" in df.columns:
    fare_median = df["Fare"].median()
    df["Fare"] = df["Fare"].fillna(fare_median)

    print("\nMissing Fare values after median imputation:")
    print(df["Fare"].isnull().sum())


# ============================================================
# 8. HANDLE MISSING CABIN
# ============================================================

if "Cabin" in df.columns:
    df["Cabin"] = df["Cabin"].fillna("Unknown")


# ============================================================
# 9. LABEL ENCODING - SEX
# ============================================================

label_encoder = LabelEncoder()

df["Sex_Encoded"] = label_encoder.fit_transform(df["Sex"])

print("\n" + "=" * 60)
print("LABEL ENCODING - SEX")
print("=" * 60)

print(df[["Sex", "Sex_Encoded"]].head(10))


# ============================================================
# 10. ONE-HOT ENCODING - EMBARKED
# ============================================================

embarked_encoded = pd.get_dummies(
    df["Embarked"],
    prefix="Embarked",
    dtype=int
)

df = pd.concat(
    [df, embarked_encoded],
    axis=1
)

print("\n" + "=" * 60)
print("ONE-HOT ENCODING - EMBARKED")
print("=" * 60)

print(df.head())


# ============================================================
# 11. AGE VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Age"],
    bins=20,
    kde=True
)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.show()


# ============================================================
# 12. SURVIVAL VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Survived"
)

plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.show()


# ============================================================
# 13. MACHINE LEARNING
# ============================================================

print("\n" + "=" * 60)
print("MACHINE LEARNING")
print("=" * 60)


# Features used for prediction
features = [
    "Pclass",
    "Sex_Encoded",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_C",
    "Embarked_Q",
    "Embarked_S"
]

X = df[features]

# Target variable
y = df["Survived"]


# ============================================================
# 14. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 15. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# ============================================================
# 16. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 17. MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL ACCURACY")
print("=" * 60)

print("Accuracy:", accuracy)
print("Accuracy Percentage:", round(accuracy * 100, 2), "%")


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)

print(cm)


# ============================================================
# 19. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(classification_report(y_test, y_pred))


# ============================================================
# 20. NEW PASSENGER PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("NEW PASSENGER PREDICTION")
print("=" * 60)

new_passenger = pd.DataFrame({
    "Pclass": [3],
    "Sex_Encoded": [1],
    "Age": [25],
    "SibSp": [0],
    "Parch": [0],
    "Fare": [10],
    "Embarked_C": [0],
    "Embarked_Q": [0],
    "Embarked_S": [1]
})

prediction = model.predict(new_passenger)

if prediction[0] == 1:
    print("Prediction: Passenger is likely to SURVIVE")
else:
    print("Prediction: Passenger is likely to NOT SURVIVE")


# ============================================================
# 21. FINAL MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("FINAL MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())


# ============================================================
# 22. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    "cleaned_titanic.csv",
    index=False
)

print("\n" + "=" * 60)
print("WEEK 1 PROJECT COMPLETED")
print("=" * 60)

print("\nCleaned dataset saved as:")
print("cleaned_titanic.csv")