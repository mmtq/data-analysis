import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
data = pd.read_csv('ecommerce_customer_behavior_dataset.csv')
# print(data.head())

# -------------------------------Level 1: Basic Insights--------------------------------


# -----------------------------Q.1 Finding Mean, Median, and Mode (Age)------------------------------
mean_age = data['Age'].mean()
median_age = data['Age'].median()
mode_age = data['Age'].mode()

print(f"Mean Age: {mean_age}, Median Age: {median_age}, Mode Age: {mode_age}")

# -----------------------------Q.2 Finding Variance and Standard Deviation (Purchase Amount)------------------------------

variance_purchase = data['Purchase Amount ($)'].var()
standard_deviation_purchase = data['Purchase Amount ($)'].std()

print(f"Variance: {variance_purchase}, Standard Deviation: {standard_deviation_purchase}")

data['Z-Score Purchase'] = (data['Purchase Amount ($)'] - data['Purchase Amount ($)'].mean()) / data['Purchase Amount ($)'].std()

print(data[['Purchase Amount ($)', 'Z-Score Purchase']].head())

sns.scatterplot(x='Purchase Amount ($)', y='Z-Score Purchase', data=data)
plt.title('Z-Score Purchase')
plt.xlabel('Purchase Amount ($)')
plt.ylabel('Z-Score Purchase')
plt.show()

# -----------------------------Q.3: Top Three Product Categories Based on Number of Purchases------------------------------

top_product_categories = data['Product Category'].value_counts().nlargest(3)
print(f"Top 3 Product Categories:\n{top_product_categories}")

plt.figure(figsize=(8, 6))
sns.barplot(x=top_product_categories.index, y=top_product_categories.values, palette="Greens_d")
plt.title('Top 3 Product Categories')
plt.xlabel('Product Category')
plt.ylabel('Number of Purchases')
plt.show()

# -----------------------------Q.4: Number of Return Customers------------------------------


return_customers_count = data['Return Customer'].sum()
print(f"Number of Return Customers: {return_customers_count}")

# ------------------------------Q.5: Average Review Score-----------------------------

average_review_score = data['Review Score (1-5)'].mean()
print(f"Average Review Score: {average_review_score}")

# -----------------------------Q.6: Average Delivery Time by Subscription Status-----------------------------

avg_delivery_time_by_subscription = data.groupby('Subscription Status')['Delivery Time (days)'].mean()
print(f"Average Delivery Time by Subscription Status:\n{avg_delivery_time_by_subscription}")

plt.figure(figsize=(8, 6))
sns.barplot(x=avg_delivery_time_by_subscription.index, y=avg_delivery_time_by_subscription.values, palette="Oranges_d", hue_norm=None)
plt.title('Average Delivery Time by Subscription Status')
plt.xlabel('Subscription Status')
plt.ylabel('Average Delivery Time (days)')
plt.yticks(np.arange(0, 8, .25))
plt.show()

# -----------------------------Q.7: Number of Subscribed Customers without Trial-----------------------------

subscribed_customers_count = len(data[data['Subscription Status'] != 'Trial'])
print(f"Number of Subscribed Customers: {subscribed_customers_count}")

# -----------------------------Q.8: Device Usage Percentages-----------------------------

# Calculate the percentage of customers using each device type
device_usage_percent = data['Device Type'].value_counts(normalize=True) * 100
print(f"Device Usage Percentages:\n{device_usage_percent}")

plt.figure(figsize=(8, 6))
sns.barplot(x=device_usage_percent.index, y=device_usage_percent.values, palette="Blues_d")
plt.title('Device Usage Percentages')
plt.xlabel('Device Type')
plt.ylabel('Percentage (%)')
plt.show()

# -----------------------------Q.9: Purchase Amount with and without discounts-----------------------------

avg_purchase_discount = data[data['Discount Availed'] == True]['Purchase Amount ($)'].mean()
avg_purchase_no_discount = data[data['Discount Availed'] == False]['Purchase Amount ($)'].mean()

print(f"Average Purchase Amount (With Discount): {avg_purchase_discount}")
print(f"Average Purchase Amount (Without Discount): {avg_purchase_no_discount}")

# -----------------------------Q.10: Most Common Payment Method-----------------------------

most_common_payment_method = data['Payment Method'].mode()[0]
print(f"Most Common Payment Method: {most_common_payment_method}")


# -----------------------------Level 2: Intermediate Insights--------------------------------


# -----------------------------Q.1: Average Review Score for User with Most Common Payment Method-----------------------------
average_review_score_user_most_common_payment_method = data[data['Payment Method'] == most_common_payment_method]['Review Score (1-5)'].mean()
print(f"Average Review Score for User with Most Common Payment Method: {average_review_score_user_most_common_payment_method}")

# -----------------------------Q.2: Correlation between Time Spent on Website and Purchase Amount-----------------------------

correlation = data['Time Spent on Website (min)'].corr(data['Purchase Amount ($)'])
print(f"Correlation: {correlation}")

# plt.scatter(data['Time Spent on Website (min)'], data['Purchase Amount ($)'], alpha=0.1, color='green') #alternative way
sns.scatterplot(x='Time Spent on Website (min)', y='Purchase Amount ($)', data=data, alpha=0.6)
plt.title('Time Spent on Website vs Purchase Amount')
plt.xlabel('Time Spent on Website (minutes)')
plt.ylabel('Purchase Amount ($)')
plt.show()


grouped_data = data.groupby('Time Spent on Website (min)')['Purchase Amount ($)'].mean()

print(grouped_data)

plt.figure(figsize=(10, 6))
sns.lineplot(x=grouped_data.index, y=grouped_data.values)
plt.title('Time Spent on Website vs Average Purchase Amount')
plt.xlabel('Time Spent on Website (min)')
plt.ylabel('Average Purchase Amount ($)')
plt.show()


# -----------------------------Q.3: Percentage of Satisfied Return Customers-----------------------------

satisfied_customers = data[(data['Review Score (1-5)'] >= 4)]
return_customers = satisfied_customers[satisfied_customers['Return Customer'] == True]
percentage = (len(return_customers) / len(data)) * 100
print(f"Percentage of satisfied return customers: {percentage:.2f}%")

# -----------------------------Q.4: Number of Items Purchased vs Customer Satisfaction-----------------------------

purchased_items_vs_satisfaction = data.groupby('Number of Items Purchased')['Review Score (1-5)'].mean()

purchased_items_vs_satisfaction.plot(kind='bar')
# sns.barplot(x='Number of Items Purchased', y='Review Score (1-5)', data=data)
plt.title('Number of Items Purchased vs Customer Satisfaction')
plt.xlabel('Number of Items Purchased')
plt.ylabel('Average Customer Satisfaction Rating')
plt.show()

items_satisfaction = data.groupby('Customer Satisfaction')['Number of Items Purchased'].mean()

print(items_satisfaction)


plt.figure(figsize=(8, 6))
sns.boxplot(x='Customer Satisfaction', y='Number of Items Purchased', data=data, order=['Low', 'Medium', 'High'])
plt.title('Number of Items Purchased vs Customer Satisfaction')
plt.xlabel('Customer Satisfaction')
plt.ylabel('Number of Items Purchased')
plt.show()


# -----------------------------Q.5: 2nd Highest Average Purchase Amount by Location-----------------------------

location_avg_purchase = data.groupby('Location')['Purchase Amount ($)'].mean()

sorted_locations = location_avg_purchase.sort_values(ascending=False)

second_highest_location = sorted_locations.index[1]
second_highest_avg_purchase = sorted_locations.iloc[1]
print(f"Location with the 2nd highest average purchase amount: {second_highest_location} (${second_highest_avg_purchase:.2f})")


