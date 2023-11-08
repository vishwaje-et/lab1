![image](https://github.com/vishwaje-et/lab1/assets/110335323/02a26121-aad5-41a1-921c-0e724ff0340c)Introduction:
Principal Component Analysis (PCA) is a powerful technique used for dimensionality reduction by transforming highdimensional data into a lower-dimensional space while retaining as much variance as possible.
Objective:
The objective of this lab is to utilize the PCA algorithm to reduce the dimensionality of the wine dataset while
preserving the most significant variations in the data. The aim is to simplify the dataset representation and improve
the distinction between red and white wines using the transformed principal components.
Dataset: Wine.csv
TheoryStep 1: Importing Libraries and Loading the Dataset
1. Open your preferred Python environment (such as Jupyter Notebook).
2. Import the necessary libraries: pandas, numpy, matplotlib.pyplot, and sklearn.
3. Load the dataset using pandas from the provided CSV link.
Step 2: Data Preprocessing
1. Check the basic information about the dataset using the `info()` function.
2. Separate the features (variables) and the target variable (wine type) from the dataset.
3. Standardize the features to have zero mean and unit variance using the `StandardScaler` from sklearn.
Step 3: Applying PCA Algorithm
1. Import the `PCA` class from sklearn.
2. Create an instance of the `PCA` class.
3. Fit the PCA model to the standardized data using the `fit()` method.
4. Explore the explained variance ratio for each principal component using the `explained_variance_ratio_` attribute.
5. Plot a cumulative explained variance plot to visualize how much variance is captured by a certain number of
principal components.
Step 4: Choosing the Number of Principal Components
1. Based on the cumulative explained variance plot, decide how many principal components to retain for
dimensionality reduction.
2. Retrain the PCA model with the selected number of components.
Step 5: Transforming Data and Visualization
1. Transform the standardized data using the `transform()` method of the PCA model.
2. Create a new DataFrame with the transformed data and assign appropriate column names.
3. Visualize the transformed data using scatter plots, with the principal components as axes.
Step 6: Wine Classification using Principal Components
1. Split the transformed data and the target variable (wine type) into training and testing sets.
2. Choose a classification algorithm (e.g., Logistic Regression, Random Forest) and train it using the transformed data.
3. Evaluate the classification model's performance on the testing set.
Conclusion:
In this lab, we have successfully applied the PCA algorithm to the wine dataset, reducing its dimensionality while
retaining significant variations in the data. By visualizing the transformed data and applying a classification algorithm,
we have demonstrated how the reduced set of principal components can help distinguish between red and white
wines more effectively than the original high-dimensional data. This technique showcases the power of
dimensionality reduction in simplifying complex datasets while preserving meaningful information.

 Fig 1 Principal Component Analysis
Assignment 2
Regression Analysis for Predicting Uber Ride Prices
Introduction:
Regression analysis is a statistical technique used to model the relationship between a dependent variable and one
or more independent variables. In this analysis, we have predict the price of an Uber ride based on various features
such as pickup point, drop-off location, etc.
Dataset:
Dataset :Uber Fares Dataset
Analysis Steps:
Step 1: Importing Libraries and Loading the Dataset
1. Open your preferred Python environment (e.g., Jupyter Notebook).
2. Import the necessary libraries: pandas, numpy, matplotlib.pyplot, seaborn, sklearn.
3. Load the dataset using pandas from the provided CSV link.
Step 2: Data Preprocessing
1. Check basic information about the dataset using the `info()` and `head()` functions.
2. Handle missing values if any.
3. Convert categorical variables into numerical representations using techniques like one-hot encoding.
4. Normalize or standardize the numerical features if needed.
Step 3: Identifying Outliers
1. Visualize the distribution of numerical variables using box plots or histograms.
2. Identify potential outliers using appropriate techniques (e.g., IQR, Z-score).
Step 4: Checking Correlations
1. Calculate the correlation matrix of numerical features.
2. Visualize the correlation matrix using a heatmap to identify strong correlations.
Step 5: Implementing Regression Models
1. Separate the features (independent variables) and the target variable (ride price) from the dataset.
2. Split the data into training and testing sets using `train_test_split` from sklearn.
3. Implement linear regression, ridge regression, and Lasso regression models using the appropriate classes from
sklearn.
4. Fit each model to the training data.
Step 6: Evaluating Models
1. Make predictions using each model on the testing set.
2. Calculate metrics such as R-squared, Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), etc., for each
model to evaluate their performance.
3. Compare the performance metrics of the models to determine which one performs better.
Step 7: Model Comparison
1. Compare the performance of linear regression, ridge regression, and Lasso regression models.
2. Choose the model that yields the best performance based on the evaluation metrics.
3. Conclude the analysis by discussing the insights gained, model performance, and potential areas for improvement.

 Fig 2 Regression Model
Conclusion:
In this analysis, you successfully performed regression analysis to predict the price of Uber rides based on various
features. By comparing the metrics of these models, we gained insights into which model provides the best
predictions for ride prices. This analysis showcases the power of regression techniques in predicting real-world
outcomes based on relevant features.

