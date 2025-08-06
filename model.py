import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib  # for saving model

# Load and prepare data
df = pd.read_csv(r'D:\WAIZ CS\PROGRAMMING\PROGRAMES using python\DATA SCIENCE\DATA SCIENCE IN JUPYTER\machine learning\DATA\HR_employees_classification.csv')  # Change this to your actual file

# Clean target
df.drop(['Department' ,'salary' ,'promotion_last_5years' ,'left' ,'Work_accident' ,'last_evaluation'] ,axis=1 ,inplace=True)

def classify(row):
    if row['satisfaction'] >= 0.64 and row['number_project'] >= 4 and row ['average_montly_hours'] >= 200 and row['time_spend_company'] >= 3:
        return 'Good'
    else: 
        return 'Bad'

df['Employee_Label'] = df.apply(classify, axis=1)
df['Employee_Label'] = df['Employee_Label'].astype(str).str.strip().str.capitalize()
df['Employee_Label'] = df['Employee_Label'].map({'Bad': 0, 'Good': 1})
df = df.dropna(subset=['Employee_Label'])

# Features and target
features = ['satisfaction', 'number_project', 'average_montly_hours', 'time_spend_company']
X = df[features]
y = df['Employee_Label']

# Scale and split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'employee_classifier.pkl')
joblib.dump(scaler, 'scaler.pkl')
