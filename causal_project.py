import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

dates = pd.date_range(start="2023-05-01", end="2023-07-31")

sales = np.random.normal(loc=200, scale=20, size=len(dates))

sales = [s + 40 if d >= pd.Timestamp("2023-06-01") else s for s, d in zip(sales, dates)]

df = pd.DataFrame({
    "date": dates,
    "sales": sales
})

df["price"] = 1000
df["discount"] = df["date"].apply(lambda x: 1 if x >= pd.Timestamp("2023-06-01") else 0)

df['date'] = pd.to_datetime(df['date'])

intervention_date = pd.Timestamp("2023-06-01")

df['treatment'] = (df['date'] >= intervention_date).astype(int)

df['post'] = df['treatment']
df['interaction'] = df['treatment'] * df['post']

# Final check
print(df.columns)
print(df.head()) 

import statsmodels.api as sm

X = df[['treatment']]
X = sm.add_constant(X)

y = df['sales']

model = sm.OLS(y, X).fit()

print(model.summary())

impact = model.params['treatment']

print("\nCausal Impact Analysis:")
print("----------------------")

if impact > 0:
    print(f"✅ Discount increased sales by approx {round(impact,2)} units")
else:
    print("❌ Discount did not improve sales")




