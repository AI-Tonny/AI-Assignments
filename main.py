import numpy as np
import pandas as pd

def generate_data():
    np.random.seed(42)
    number_of_samples = 2000

    data = {
        'customer_id': range(1, number_of_samples + 1),
        'age': np.random.randint(18, 70, number_of_samples),
        'income': np.random.randint(20000, 150000, number_of_samples),
        'credit_score': np.random.randint(300, 850, number_of_samples),
        'account_balance': np.random.randint(-5000, 100000, number_of_samples),
        'num_products': np.random.randint(1, 5, number_of_samples),
        'has_credit_card': np.random.choice([0, 1], number_of_samples),
        'is_active_member': np.random.choice([0, 1], number_of_samples),
        'country': np.random.choice(['Ukraine', 'Poland', 'Germany'], number_of_samples),
        'churn': np.random.choice([0, 1], number_of_samples, p=[0.7, 0.3]),
        'tenure': np.random.randint(0, 11, number_of_samples)
    }

    df = pd.DataFrame(data)
    return df

df = generate_data()

# print("First ten rows:\n", df.head(10))
# print("Last ten rows:\n", df.tail(10))

# df.info()
# print(df.describe())

# df_target = df[(df['country'] == 'Germany') & (df['age'] > 30) & (df['account_balance'] > 0)]
# print(df_target.head(10))

# df.loc[np.random.choice(range(2000), size=200, replace=False), 'num_products'] = np.nan
# df.loc[(df['num_products'].isna()) & (df['is_active_member'] == 1), 'num_products'] = 2
# df.loc[(df['num_products'].isna()) & (df['is_active_member'] == 0), 'num_products'] = 1

# df = pd.concat([df, df.head(50)], ignore_index=True)
# df = df.drop_duplicates(keep='first')

# df['credit_category'] = pd.cut(df['credit_score'], bins=[299, 500, 700, 850], labels=['Poor', 'Fair', 'Good'])

# summary = df.groupby(['country', 'age'])[['income', 'account_balance']].mean()
# pivot = summary.unstack()
#
# print(pivot)

# churn_corr = df.corr(numeric_only=True)['churn']
# churn_corr = churn_corr.drop('churn')
#
# strongest_feature = churn_corr.abs().idxmax()
# correlation_value = churn_corr[strongest_feature]
#
# print(strongest_feature, correlation_value)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def prepare_data(df):
    X = df[['credit_score', 'age', 'account_balance', 'income']]
    y = df['churn']

    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = prepare_data(df)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Точність моделі: {accuracy_score(y_test, y_pred):.2%}")
print("\nДетальний звіт:")
print(classification_report(y_test, y_pred))

df['churn_probability'] = model.predict_proba(df[['credit_score', 'age', 'account_balance', 'income']])[:, 1]
top_risk_customers = df[df['churn'] == 0].sort_values(by='churn_probability', ascending=False).head(5)

print("\nТоп-5 клієнтів які можуть піти:")
print(top_risk_customers[['credit_score', 'age', 'account_balance', 'income', 'churn_probability']])