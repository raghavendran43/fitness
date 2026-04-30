import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
df = pd.read_csv(r"C:\Users\ragha\Downloads\ts_transport_fitness_01_11_2025to30_11_2025.csv")
print(df.columns)
df['validTo'] = pd.to_datetime(df['validTo'], dayfirst=True, errors='coerce')
df['validFrom'] = pd.to_datetime(df['validFrom'], errors='coerce')
df['validTo'] = pd.to_datetime(df['validTo'], errors='coerce')

df['fromdate'] = pd.to_datetime(df['fromdate'], errors='coerce')

fitness_counts = df['status'].value_counts()
print(fitness_counts)

fitness_counts.plot(kind='bar', title="Vehicle Fitness Status")
plt.show()

today = pd.Timestamp.today()

expired = df[df['validTo'] < today]
due_soon = df[(df['validTo'] >= today) & (df['validTo'] <= today + pd.Timedelta(days=30))]

print(len(expired))
print(len(due_soon))

category_analysis = pd.crosstab(df['vehicleClass'], df['status'])
print(category_analysis)

category_analysis.plot(kind='bar', stacked=True, title="Vehicle Class vs Fitness")
plt.show()

df['issue_month'] = df['validFrom'].dt.to_period('M')

trend = df.groupby('issue_month').size()
print(trend)

trend.plot(title="Fitness Certificates Over Time")
plt.show()

region_analysis = df['OfficeCd'].value_counts().head(10)
print(region_analysis)

region_analysis.plot(kind='bar', title="Top Offices")
plt.show()

owner_counts = df['registrationNo'].value_counts().head(10)
print(owner_counts)

owner_counts.plot(kind='bar', title="Top Vehicles")
plt.show()

current_year = pd.Timestamp.today().year
df['vehicle_age'] = current_year - df['fromdate'].dt.year

age_vs_fitness = df.groupby('vehicle_age')['status'].value_counts().unstack()
print(age_vs_fitness)

age_vs_fitness.plot(kind='bar', stacked=True, title="Vehicle Age vs Fitness")
plt.show()

df['validity_days'] = (df['validTo'] - df['validFrom']).dt.days

print(df['validity_days'].describe())

df['validity_days'].plot(kind='hist', bins=20, title="Validity Distribution")
plt.show()

ml_df = df[['vehicle_age', 'validity_days']].dropna()

X = ml_df[['vehicle_age']]
y = ml_df['validity_days']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
print(model.coef_[0])
print(model.intercept_)

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.title("Vehicle Age vs Validity")
plt.xlabel("Vehicle Age")
plt.ylabel("Validity Days")
plt.show()
