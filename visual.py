import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

train_dataset = pd.read_csv('dataset/train.csv')

encoder = LabelEncoder()
train_dataset['IsHoliday'] = encoder.fit_transform(train_dataset['IsHoliday'])
train_dataset['Date'] = pd.to_datetime(train_dataset['Date'])
train_dataset['Year'] = train_dataset['Date'].dt.year
train_dataset['Month'] = train_dataset['Date'].dt.month
train_dataset['Day'] =  train_dataset['Date'].dt.day
train_dataset.drop(columns=['Date'], inplace=True)

feature_dataset = pd.read_csv('dataset/features.csv')
feature_dataset.dropna(inplace=True)
feature_dataset['IsHoliday'] = encoder.fit_transform(feature_dataset['IsHoliday'])
feature_dataset['Date'] = pd.to_datetime(feature_dataset['Date'])
feature_dataset['Year'] = feature_dataset['Date'].dt.year
feature_dataset['Month'] = feature_dataset['Date'].dt.month
feature_dataset['Day'] =  feature_dataset['Date'].dt.day
feature_dataset.drop(columns=['Date'], inplace=True)

joined_dataset = pd.merge(train_dataset,feature_dataset, on=['Year','Month','Day','Store','IsHoliday'], how='inner')
joined_dataset.to_csv('joined_dataset')
'''

w_sle_store = joined_dataset.groupby('Store')['Weekly_Sales'].sum().reset_index().rename(columns={'Weekly_Sales': 'Total_Sales'}).sort_values(by='Total_Sales', ascending=False).head(10)
plt.bar(w_sle_store['Store'].values.astype(str),w_sle_store['Total_Sales'].values)
plt.xlabel('Stores')
plt.ylabel('Weekly Sales')
plt.title('Highest Weekly Sales per Store ( Top 10 )')
plt.show()

w_sle_dept = joined_dataset.groupby('Dept')['Weekly_Sales'].mean().reset_index().rename(columns={'Weekly_Sales': 'Total_Sales'}).sort_values(by='Total_Sales', ascending=False).head(10)
plt.bar(w_sle_dept['Dept'].values.astype(str),w_sle_dept['Total_Sales'].values)
plt.xlabel('Department')
plt.ylabel('Weekly Sales')
plt.title('Highest Weekly Sales per Department ( Top 10 )')
plt.show()

yr_mn_sales = (joined_dataset.groupby(['Year', 'Month'])['Weekly_Sales'].sum().reset_index().rename(columns={'Weekly_Sales': 'Total_Sales'})).sort_values(by='Total_Sales', ascending=False)
plt.figure(figsize=(10, 6))
yr_mn_sales['Year-Month'] = yr_mn_sales['Year'].astype(str) + '-' + yr_mn_sales['Month'].apply(lambda x: f'{x:02}')
plt.plot(yr_mn_sales['Year-Month'], yr_mn_sales['Total_Sales'], marker='o', linestyle='-', color='b')

plt.xlabel('Year-Month')
plt.ylabel('Total Sales')
plt.title('Total Sales Over Time')
plt.xticks(rotation=45) 
plt.tight_layout()

plt.show()

markdown_sales = joined_dataset.groupby(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'])['Weekly_Sales'].sum().reset_index()

markdown_sales_melted = markdown_sales.melt(id_vars=['Weekly_Sales'], var_name='Markdown_Type', value_name='Total_Sales')

plt.figure(figsize=(14, 8))

for markdown_type in markdown_sales_melted['Markdown_Type'].unique():
    subset = markdown_sales_melted[markdown_sales_melted['Markdown_Type'] == markdown_type]
    plt.plot(subset['Markdown_Type'], subset['Total_Sales'], marker='o', label=markdown_type)

plt.xlabel('Markdown Types')
plt.ylabel('Total Weekly Sales')
plt.title('Impact of Each Markdown on Weekly Sales')

plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(title='Markdown Type')
plt.tight_layout()
plt.show()

'''







