# FTDE-2 Homework Machine Learning

## Project Description
This project is part of the Fast Track Data Engineering program from Digital Skola Batch 2. The goal of this project is to build a machine learning model using the provided dataset. Specifically, the objective is to predict whether a user is interested in purchasing vehicle insurance. 
By completing this project, we aim to understand the fundamental concepts of machine learning in a simplified manner. The machine learning algorithm used in this project is Random Forest.

## About Dataset
The dataset contains insurance-related information. There are 12 features, with one target feature, Response. The data is provided in CSV format, which can be found in the /data folder. Below is a description of the dataset features:

* id: Unique identifier for each entry or customer.
* Gender: The customer's gender ('Male' or 'Female').
* Age: The customer's age in years.
* Driving_License: Indicates whether the customer holds a driving license (0 for no, 1 for yes).
* Region_Code: A numeric code representing the geographical area where the customer resides.
* Previously_Insured: Indicates whether the customer already holds an insurance policy (0 for no, 1 for yes).
* Vehicle_Age: Age of the customer's vehicle ('New', '1-2 Years', 'More than 2 Years').
* Vehicle_Damage: Indicates whether the customer's vehicle has been damaged ('Yes' or 'No').
* Annual_Premium: The annual insurance premium paid by the customer (in the relevant currency).
* Policy_Sales_Channel: A numerical code representing the sales channel (e.g., agent, online, etc.).
* Vintage: The number of days since the customer first interacted with the insurance company.
* Response: The target variable indicating the customer's interest in the insurance offer (0 for not interested, 1 for interested).

* ## Workflow & Results
The trained machine learning model is saved in .pkl format (Pickle). Due to file size limitations, you can generate the model by running the Homework.ipynb notebook or loading model_rf_insurance.

The Random Forest model achieved an accuracy of 86.67% on the test set. 

![image](https://github.com/user-attachments/assets/cef6ddb5-c42c-4a08-b9a8-093ae8c3b5cd)



## Future Improvements
1. Feature Engineering:
   * Perform feature engineering on categorical features such as Policy_Sales_Channel and Region_Code. Convert these into categorical variables using techniques like One-Hot Encoding or Label Encoding.

2. Data Cleaning:
   * Investigate and handle any missing values or outliers in the dataset, as they can significantly affect model performance.

3. Hyperparameter Tuning:
   * Implement hyperparameter tuning using techniques like Grid Search or Random Search to optimize the Random Forest model parameters.

4. Cross-Validation:
   * Use cross-validation to evaluate the model’s performance more reliably and ensure that it generalizes well to unseen data.

5. Model Comparison:
   * Compare the Random Forest model with other algorithms, such as Logistic Regression, Gradient Boosting, or Support Vector Machines, to identify the best-performing model for this task.

and etc.
