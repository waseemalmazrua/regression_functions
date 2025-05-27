# 📦 جميع المكتبات المطلوبة
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    r2_score, mean_squared_error
)
from sklearn import set_config
import pandas as pd
import numpy as np

# لعرض نتائج الـ Pipeline بشكل منظم
set_config(transform_output="pandas")

# 📌 Logistic Regression Pipeline

def run_pipeline_logistic_regression(df, target_col, test_size=0.2, random_state=42):
    """
    تدريب نموذج Logistic Regression باستخدام Pipeline.
    يشمل: OneHotEncoder للفئوية، StandardScaler للرقمية.
    يقبل أي DataFrame ويحدد الأعمدة تلقائيًا.
    """
    # فصل المتغير الهدف عن الميزات
    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    # ترميز الفئة المستهدفة إن كانت نصية
    if y_raw.dtype == 'object':
        y = y_raw.map({val: i for i, val in enumerate(y_raw.unique())})
    else:
        y = y_raw

    # تحديد الأعمدة
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # تجهيز الـ ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ])

    # إنشاء الـ Pipeline
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # تدريب النموذج
    pipeline.fit(X_train, y_train)

    # التنبؤ
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # تقييم الأداء
    print("\n📊 Logistic Regression Results")
    print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Test Accuracy :", accuracy_score(y_test, y_test_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

    return pipeline


# 📌 Linear Regression Pipeline

def run_pipeline_linear_regression(df, target_col, test_size=0.2, random_state=42):
    """
    تدريب نموذج Linear Regression باستخدام Pipeline.
    يشمل: OneHotEncoder للفئوية، StandardScaler للرقمية.
    يقبل أي DataFrame ويحدد الأعمدة تلقائيًا.
    """
    # فصل المتغير الهدف عن الميزات
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # تحديد الأعمدة
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # تجهيز الـ ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ])

    # إنشاء الـ Pipeline
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('regressor', LinearRegression())
    ])

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # تدريب النموذج
    pipeline.fit(X_train, y_train)

    # التنبؤ
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # تقييم الأداء
    print("\n📈 Linear Regression Results")
    print("Train R² Score:", r2_score(y_train, y_train_pred))
    print("Test R² Score :", r2_score(y_test, y_test_pred))
    print("Test Mean Squared Error:", mean_squared_error(y_test, y_test_pred))

    return pipeline
