import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = pd.read_csv('ecommerce_customer_behavior_dataset.csv')

# # avg_delivery_time_for_returning_customers = data[data['Return Customer'] == True]['Delivery Time (days)'].mean()
# # print(f"Average Delivery Time for Returning Customers: {avg_delivery_time_for_returning_customers:.2f} days")

# # avg_delivery_time_for_non_returning_customers = data[data['Return Customer'] == False]['Delivery Time (days)'].mean()
# # print(f"Average Delivery Time for Non Returning Customers: {avg_delivery_time_for_non_returning_customers:.2f} days")


# # Q.1: 
# # Select features and target
# X = data[['Purchase Amount ($)', 'Time Spent on Website (min)', 'Review Score (1-5)', 'Number of Items Purchased', 'Delivery Time (days)', 'Customer Satisfaction Numerical']]  # Add other relevant features
# y = data['Return Customer']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the logistic regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Get the coefficients and the feature names
# coefficients = model.coef_[0]
# features = X.columns

# # Create a DataFrame to show each feature's coefficient
# feature_importance = pd.DataFrame({
#     'Feature': features,
#     'Coefficient': coefficients
# })

# # Sort by absolute value of the coefficient to see the most influential features
# feature_importance['Absolute_Coefficient'] = np.abs(feature_importance['Coefficient'])
# feature_importance = feature_importance.sort_values(by='Absolute_Coefficient', ascending=False)

# print(feature_importance)

# plt.figure(figsize=(8, 6))
# plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color='skyblue')
# plt.xlabel('Coefficient Value')
# plt.title('Feature Importance in Predicting Return Customers')
# plt.axvline(0, color='grey', linestyle='--')  # Add a vertical line at x=0 for reference
# plt.show()


# # satisfaction_payment_method = data.groupby(['Payment Method', 'Customer Satisfaction']).size().unstack()
# # print(satisfaction_payment_method)

# avg_review_by_payment = data.groupby('Payment Method')['Customer Satisfaction Numerical'].mean()

# # 2. Group by 'Payment Method' and calculate the return rate (percentage of return customers)
# return_rate_by_payment = data.groupby('Payment Method')['Return Customer'].mean() * 100

# # Create DataFrames for exporting
# avg_review_df = avg_review_by_payment.reset_index()
# avg_review_df.columns = ['Payment Method', 'Average Review Score']

# return_rate_df = return_rate_by_payment.reset_index()
# return_rate_df.columns = ['Payment Method', 'Return Rate (%)']

# # Export to CSV
# avg_review_df.to_csv('avg_review_by_payment.csv', index=False)
# return_rate_df.to_csv('return_rate_by_payment.csv', index=False)

# # Display the results
# print("Average Review Score by Payment Method:")
# print(avg_review_df)

# print("\nReturn Rate by Payment Method (%):")
# print(return_rate_df)

# # 3. Visualization of Payment Method vs Customer Satisfaction and Return Rates
# plt.figure(figsize=(14, 6))

# # Plot 1: Payment Method vs Average Review Score (Customer Satisfaction)
# plt.subplot(1, 2, 1)
# sns.barplot(x='Payment Method', y='Average Review Score', data=avg_review_df, palette="Blues_d")
# plt.title('Payment Method vs Customer Satisfaction (Average Review Score)')
# plt.xlabel('Payment Method')
# plt.ylabel('Average Review Score')

# # Plot 2: Payment Method vs Return Rate (%)
# plt.subplot(1, 2, 2)
# sns.barplot(x='Payment Method', y='Return Rate (%)', data=return_rate_df, palette="Greens_d")
# plt.title('Payment Method vs Return Rate (%)')
# plt.xlabel('Payment Method')
# plt.ylabel('Return Rate (%)')

# plt.tight_layout()
# plt.show()


# 1. Group by location and calculate the average purchase amount and delivery time
# location_purchase_delivery = data.groupby('Location')[['Purchase Amount ($)', 'Delivery Time (days)']].mean()

# # Print the average purchase amount and delivery time for each location
# print(location_purchase_delivery)

# location_purchase_delivery_csv = location_purchase_delivery.reset_index()
# location_purchase_delivery_csv.columns = ['Location', 'Purchase Amount ($)', 'Delivery Time (days)']

# location_purchase_delivery_csv.to_csv('location_purchase_delivery.csv', index=False)

# 2. Visualize the average purchase amount by location
# plt.figure(figsize=(10, 6))
# sns.barplot(x=location_purchase_delivery.index, y=location_purchase_delivery['Purchase Amount ($)'])
# plt.title('Average Purchase Amount by Location')
# plt.xlabel('Location')
# plt.ylabel('Average Purchase Amount')
# plt.xticks(rotation=90)  # Rotate the x-axis labels if there are many locations
# plt.show()

# # 3. Visualize the average delivery time by location
# plt.figure(figsize=(10, 6))
# sns.barplot(x=location_purchase_delivery.index, y=location_purchase_delivery['Delivery Time (days)'])
# plt.title('Average Delivery Time by Location')
# plt.xlabel('Location')
# plt.ylabel('Average Delivery Time (in days)')
# plt.xticks(rotation=90)
# plt.show()


device_satisfaction = data.groupby(['Device Type', 'Customer Satisfaction']).size().unstack()

# Normalize the counts to get proportions within each device category
device_satisfaction_normalized = device_satisfaction.div(device_satisfaction.sum(axis=1), axis=0)

# Print the satisfaction distribution for each device
# print(device_satisfaction_normalized)

# 2. Visualize the relationship with a stacked bar plot
# device_satisfaction_normalized.plot(kind='bar', stacked=True, figsize=(10, 6))
# plt.title('Customer Satisfaction Distribution by Device Used')
# plt.xlabel('Device Used')
# plt.ylabel('Proportion of Satisfaction Levels')
# plt.legend(title='Customer Satisfaction')
# plt.xticks(rotation=0)
# plt.show()

# 3. Optional: Bar plot showing average satisfaction score for each device
# Assign numerical values to satisfaction levels (Low: 1, Medium: 2, High: 3) to calculate an average score
# satisfaction_map = {'Low': 1, 'Medium': 2, 'High': 3}
# data['satisfaction_score'] = data['Customer Satisfaction'].map(satisfaction_map)

# # Calculate average satisfaction score by device
# avg_satisfaction_by_device = data.groupby('Device Type')['satisfaction_score'].mean()

# print(avg_satisfaction_by_device)

# avg_satisfaction_by_device_csv = avg_satisfaction_by_device.reset_index()
# avg_satisfaction_by_device_csv.columns = ['Device Type', 'Average Satisfaction Score']

# avg_satisfaction_by_device_csv.to_csv('avg_satisfaction_by_device.csv', index=False)

# Bar plot of average satisfaction scores
# plt.figure(figsize=(8, 6))
# sns.barplot(x=avg_satisfaction_by_device.index, y=avg_satisfaction_by_device.values)
# plt.title('Average Customer Satisfaction Score by Device Used')
# plt.xlabel('Device Used')
# plt.ylabel('Average Satisfaction Score')
# plt.xticks(rotation=0)
# plt.show()

# Create a binary target variable for Premium users (Premium = 1, Free/Trial = 0)
# data['is_premium'] = data['Subscription Status'].apply(lambda x: 1 if x == 'Premium' else 0)

# # Convert "Customer Satisfaction" to numerical values (Low = 1, Medium = 2, High = 3)
# satisfaction_map = {'Low': 1, 'Medium': 2, 'High': 3}
# data['satisfaction_score'] = data['Customer Satisfaction'].map(satisfaction_map)

# # Select relevant features and target (is_premium column)
# X = data[['Delivery Time (days)', 'satisfaction_score']]
# y = data['is_premium']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 2. Train a logistic regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # 3. Get the coefficients of the model
# coefficients = model.coef_[0]
# features = X.columns

# # Create a DataFrame to show each feature's coefficient
# feature_importance = pd.DataFrame({
#     'Feature': features,
#     'Coefficient': coefficients
# })

# # Sort by absolute value of the coefficient to see the most influential features
# feature_importance['Absolute_Coefficient'] = np.abs(feature_importance['Coefficient'])
# feature_importance = feature_importance.sort_values(by='Absolute_Coefficient', ascending=False)

# print(feature_importance)

# premium_customers_satisfaction = feature_importance.reset_index()
# premium_customers_satisfaction.columns = ['id', 'Feature', 'Coefficient', 'Absolute_Coefficient']
# premium_customers_satisfaction.to_csv('premium_customers_satisfaction.csv', index=False)

# premium_customers_avg_satisfaction = data[data['is_premium'] == 1]['satisfaction_score'].mean()
# print(f"Average Satisfaction Score for Premium Users: {premium_customers_avg_satisfaction:.2f}")

# non_premium_customers_avg_satisfaction = data[data['is_premium'] == 0]['satisfaction_score'].mean()
# print(f"Average Satisfaction Score for Non-Premium Users: {non_premium_customers_avg_satisfaction:.2f}")

# premium_customers_avg_delivery_time = data[data['is_premium'] == 1]['Delivery Time (days)'].mean()
# print(f"Average Delivery Time for Premium Users: {premium_customers_avg_delivery_time:.2f} days")

# non_premium_customers_avg_delivery_time = data[data['is_premium'] == 0]['Delivery Time (days)'].mean()
# print(f"Average Delivery Time for Non-Premium Users: {non_premium_customers_avg_delivery_time:.2f} days")


# plt.figure(figsize=(8,6))
# plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color='skyblue')
# # sns.barplot(x='Feature', y='Coefficient', data=feature_importance)
# plt.title('Factors Influencing Premium User Status')
# plt.xlabel('Feature')
# plt.ylabel('Coefficient')
# plt.show()


# 1. Group by Gender and Product Category, then count the occurrences
# gender_product_counts = data.groupby(['Gender', 'Product Category']).size().reset_index(name='Purchase Amount ($)')

# # 2. For each gender, find the product category with the highest count (favorite)
# favorite_category_by_gender = gender_product_counts.loc[gender_product_counts.groupby('Gender')['Purchase Amount ($)'].idxmax()]

# # Print the favorite product category for each gender
# print(favorite_category_by_gender)

# # 3. Optional: Visualize the favorite product categories by gender
# plt.figure(figsize=(8,6))
# sns.barplot(x='Gender', y='Purchase Amount ($)', hue='Product Category', data=favorite_category_by_gender)
# plt.title('Favorite Product Category by Gender')
# plt.xlabel('Gender')
# plt.ylabel('Number of Purchases')
# plt.show()

# 1. Group by Gender and Product Category, then sum the purchase amounts
gender_product_spending = data.groupby(['Gender', 'Product Category'])['Purchase Amount ($)'].sum().reset_index()

# 2. For each gender, find the product category with the highest total spending
top_spending_category_by_gender = gender_product_spending.loc[gender_product_spending.groupby('Gender')['Purchase Amount ($)'].idxmax()]

# Print the product category on which each gender spent the most
print(top_spending_category_by_gender)

top_spending_category_by_gender_csv = top_spending_category_by_gender.reset_index()
top_spending_category_by_gender_csv.columns = ['id', 'Gender', 'Product Category', 'Purchase Amount ($)']
top_spending_category_by_gender_csv.to_csv('top_spending_category_by_gender.csv', index=False)


# 3. Optional: Visualize the total spending by gender and product category

# plt.figure(figsize=(10, 6))
# sns.barplot(x='Gender', y='Purchase Amount ($)', hue='Product Category', data=top_spending_category_by_gender)
# plt.title('Product Category with the Highest Spending by Gender')
# plt.xlabel('Gender')
# plt.ylabel('Total Spending')
# plt.show()


