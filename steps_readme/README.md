# ML Service and Model registry

## Task 1

There is a  dataset containing purchase information of customers of the  some  ml service platform. Your task is to develop a model and predict how much a customer will purchase in the next month based on their previous purchases and other features.

The dataset (purchase_history.csv) includes the following columns:
```
customer_id: Unique identifier for each customer
age: Age of the customer
gender: Gender of the customer (Male/Female)
annual_income: Annual income of the customer (in USD)
purchase_amount: Total amount of the purchase (in USD)
purchase_date: Date and time of the purchase
next_month_purchase_amount: Amount predicted
```

* Choose at least two different machine learning algorithms to build predictive models.
* Discuss the results and propose one model for production usage.

Disclaimer: This dataset contains randomly generated information.

## Task 2

Implement a simple model registry service that receives a model file and saves it to a path.

This service must receive the metadata for every model uploaded and store these values in a database. Each model should have a name and version. You can define any other field as you design your data model.

This service should also expose an endpoint that returns an uploaded model according to requested model name and version.

Bonus: How would you enhance your data model? Which feature can or should be added to this application?

## Task 3

Using your selected model from Task 1 and utilizing the model registry service implemented in Task 2, create an inference service that exposes one endpoint that returns prediction result for a given single user data.


