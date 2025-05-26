from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd


def run_logistic_regression(X, y):
    # 1. تحويل الأعمدة النصية إلى متغيرات رقمية
    X_encoded = pd.get_dummies(X, drop_first=True)

    # 2. تدريب النموذج
    model = LogisticRegression(max_iter=1000)
    model.fit(X_encoded, y)

    # 3. التنبؤ على نفس البيانات
    predictions = model.predict(X_encoded)

    # 4. طباعة النتائج
    print(" Logistic Regression Results")
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)
    print("Accuracy:", accuracy_score(y, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y, predictions))
    print("Classification Report:\n", classification_report(y, predictions))

    return model


def run_linear_regression(X, y):
    
    
    يعالج الأعمدة الفئوية تلقائيًا (باستخدام one-hot encoding).
    
    Parameters:
    X -- DataFrame: المتغيرات المستقلة (قد تحتوي على أعمدة فئوية)
    y -- Series: المتغير التابع (رقمي)

    Returns:
    model -- النموذج المدرب من نوع LinearRegression
    """
    # تحويل الأعمدة الفئوية إلى أرقام
    X_encoded = pd.get_dummies(X, drop_first=True)

    model = LinearRegression()
    model.fit(X_encoded, y)
    predictions = model.predict(X_encoded)

    print("📈 Linear Regression Results")
    print("Intercept:", model.intercept_)
    print("Coefficients:")
    for feature, coef in zip(X_encoded.columns, model.coef_):
        print(f"  {feature}: {coef:.4f}")

    print("R² Score:", r2_score(y, predictions))
    print("Mean Squared Error:", mean_squared_error(y, predictions))

    return model
