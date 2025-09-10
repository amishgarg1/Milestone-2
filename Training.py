import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset from the uploaded CSV file
df = pd.read_csv('Fake Postings.csv')

# Drop rows where 'salary_range' is missing to ensure a clean dataset.
df.dropna(subset=['salary_range'], inplace=True)

# Create a function to clean and calculate the average salary from the 'salary_range' column.
def calculate_average_salary(salary_range):
    """Parses a salary range string and calculates the average salary."""
    try:
        salaries = salary_range.replace('$', '').replace(',', '').split('-')
        if len(salaries) == 2:
            return (float(salaries[0]) + float(salaries[1])) / 2
        else:
            return None
    except (ValueError, IndexError):
        return None

# Apply the function to create 'average_salary'
df['average_salary'] = df['salary_range'].apply(calculate_average_salary)

# Drop rows where 'average_salary' is None
df.dropna(subset=['average_salary'], inplace=True)

# Create dummy variables for 'employment_type'
df = pd.get_dummies(df, columns=['employment_type'], prefix='employment_type', drop_first=True)

# Define features (X) and target (y)
features = ['employment_type_Full-Time', 'employment_type_Internship', 'employment_type_Part-Time', 'employment_type_Temporary']
X = df[features]
y = df['average_salary']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print a sample of the data to match the user's output format.
# We'll display the first 5 records of X_train, y_train, X_test, y_test, and y_pred.
print("Training Data:")
print("X_train (sample):")
print(X_train.head(5))
print("\ny_train (sample):")
print(y_train.head(5))

print("\nTesting Data:")
print("X_test (sample):")
print(X_test.head(5))
print("\ny_test (sample):")
print(y_test.head(5))

print("\nPredicted Salaries (y_pred, sample):")
print(y_pred[:5])
