
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, mean_squared_error
import pandas as pd

def run_logistic_regression(X, y, test_size=0.2, random_state=42):
    """
    نموذج لوجستي مع تقسيم Train/Test.
    يعالج الأعمدة الفئوية تلقائيًا.
    """
    # ترميز الأعمدة الفئوية
    X_encoded = pd.get_dummies(X, drop_first=True)

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, random_state=random_state)

    # تدريب النموذج
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # التنبؤ على المجموعتين
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # النتائج
    print("🚀 Logistic Regression Results with Train/Test Split")
    print("Train Accuracy:", accuracy_score(y_train, train_preds))
    print("Test Accuracy :", accuracy_score(y_test, test_preds))
    print("\\nTest Confusion Matrix:\\n", confusion_matrix(y_test, test_preds))
    print("Test Classification Report:\\n", classification_report(y_test, test_preds))

    return model

def run_linear_regression(X, y, test_size=0.2, random_state=42):
    """
    نموذج انحدار خطي مع تقسيم Train/Test.
    يعالج الأعمدة الفئوية تلقائيًا.
    """
    X_encoded = pd.get_dummies(X, drop_first=True)

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, random_state=random_state)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # التنبؤ
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    print("📈 Linear Regression Results with Train/Test Split")
    print("Train R² Score:", r2_score(y_train, train_preds))
    print("Test R² Score :", r2_score(y_test, test_preds))
    print("Test Mean Squared Error:", mean_squared_error(y_test, test_preds))

    print("\\nTest Coefficients:")
    for feature, coef in zip(X_encoded.columns, model.coef_):
        print(f"  {feature}: {coef:.4f}")

    return model

