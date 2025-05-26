from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import set_config

set_config(transform_output="pandas")

def run_pipeline_logistic_regression(df, target_col, test_size=0.2, random_state=42):
    """
    تدريب نموذج Logistic Regression باستخدام Pipeline.
    تشمل المعالجة: OneHotEncoder للأعمدة النصية، StandardScaler للأعمدة الرقمية.
    """
    #  فصل الهدف عن الميزات
    X = df.drop([target_col, 'Visit_Date'], axis=1)
    y = df[target_col].map({'No': 0, 'Yes': 1})  # تحويل الفئة إلى رقم

    #  تحديد الأعمدة الرقمية والفئوية
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # ⚙️ تجهيز المعالجة باستخدام ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ])

    # إنشاء الـ Pipeline
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # ✂ تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    #  تدريب النموذج
    pipeline.fit(X_train, y_train)

    #  التنبؤ
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    #  عرض النتائج
    print(" Logistic Regression Results with Pipeline")
    print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Test Accuracy :", accuracy_score(y_test, y_test_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
    print("Classification Report:\n", classification_report(y_test, y_test_pred))

    return pipeline



from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def run_pipeline_linear_regression(df, target_col, test_size=0.2, random_state=42):
    """
    تدريب نموذج Linear Regression باستخدام Pipeline.
    تشمل المعالجة: OneHotEncoder للأعمدة النصية، StandardScaler للأعمدة الرقمية.
    """
    #  فصل الهدف عن الميزات
    X = df.drop([target_col, 'Visit_Date'], axis=1)
    y = df[target_col]

    #  تحديد الأعمدة الرقمية والفئوية
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # ⚙️ تجهيز المعالجة باستخدام ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ])

    #  إنشاء الـ Pipeline
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('regressor', LinearRegression())
    ])

    #  تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    #  تدريب النموذج
    pipeline.fit(X_train, y_train)

    #  التنبؤ
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    #  عرض النتائج
    print("📈 Linear Regression Results with Pipeline")
    print("Train R² Score:", r2_score(y_train, y_train_pred))
    print("Test R² Score :", r2_score(y_test, y_test_pred))
    print("Test Mean Squared Error:", mean_squared_error(y_test, y_test_pred))

    return pipeline
