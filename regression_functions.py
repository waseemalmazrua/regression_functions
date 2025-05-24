
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, mean_squared_error

def run_logistic_regression(X, y):
    """
    تدرب نموذج لوجستي على البيانات وتطبع ملخص النتائج.

    Parameters:
    X -- DataFrame: المتغيرات المستقلة (features)
    y -- Series: المتغير التابع (binary target 0/1)

    Returns:
    model -- النموذج المدرب من نوع LogisticRegression
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    predictions = model.predict(X)

    print("🚀 Logistic Regression Results")
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)
    print("Accuracy:", accuracy_score(y, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y, predictions))
    print("Classification Report:\n", classification_report(y, predictions))

    return model

def run_linear_regression(X, y):
    """
    تدرب نموذج انحدار خطي على البيانات وتطبع ملخص النتائج.

    Parameters:
    X -- DataFrame: المتغيرات المستقلة
    y -- Series: المتغير التابع

    Returns:
    model -- النموذج المدرب من نوع LinearRegression
    """
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    print("📈 Linear Regression Results")
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)
    print("R² Score:", r2_score(y, predictions))
    print("Mean Squared Error:", mean_squared_error(y, predictions))

    return model
