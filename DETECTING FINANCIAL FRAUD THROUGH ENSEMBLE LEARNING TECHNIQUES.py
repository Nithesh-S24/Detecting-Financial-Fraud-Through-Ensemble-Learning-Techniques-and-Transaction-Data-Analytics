#!/usr/bin/env python
# coding: utf-8

# ### Import Required Libraries

# In[1]:


# Data manipulation libraries
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# Handling imbalance
from imblearn.over_sampling import SMOTE

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


print("Libraries imported successfully")


# ### Load Dataset

# In[2]:


df = pd.read_csv("creditcard.csv")

# Display first five rows

df.head()


# ### Dataset Shape

# In[3]:


# Number of rows and columns

print("Dataset Shape:")
print(df.shape)


# ### Dataset information

# In[4]:


df.info()


# ### Statistical Summary

# In[5]:


df.describe()


# ### Check Missing Values

# In[6]:


missing_values = df.isnull().sum()

missing_values[missing_values > 0]


# ### Check Duplicate Transactions

# In[7]:


# Duplicate records

duplicates = df.duplicated().sum()

print("Number of duplicate transactions:", duplicates)


# ### Remove Duplicate Records

# In[8]:


df = df.drop_duplicates()

print("Dataset shape after removing duplicates:")
print(df.shape)


# ### Check Class Distribution

# In[9]:


class_distribution = df['Class'].value_counts()

class_distribution


# ### Visualise Fraud Distribution

# In[10]:


plt.figure(figsize=(6,4))

df['Class'].value_counts().plot(
    kind='bar'
)

plt.xlabel("Class")
plt.ylabel("Number of Transactions")
plt.title("Distribution of Fraud and Non-Fraud Transactions")

plt.show()


# ### Percentage Distribution of Fraud

# In[11]:


fraud_percentage = (
    df['Class']
    .value_counts(normalize=True)
    *100
)

fraud_percentage


# ### Transaction Amount Distribution

# In[12]:


plt.figure(figsize=(8,5))

plt.hist(
    df['Amount'],
    bins=50
)

plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.title("Transaction Amount Distribution")

plt.show()


# ### Check Correlation with Fraud Class

# In[13]:


# Correlation analysis

correlation = df.corr()

class_corr = (
    correlation['Class']
    .sort_values(ascending=False)
)

class_corr


# ### Top Features Related to Fraud

# In[14]:


plt.figure(figsize=(10,6))

class_corr.drop('Class').head(10).plot(
    kind='bar'
)

plt.title(
    "Top Features Correlated With Fraud"
)

plt.xlabel("Features")
plt.ylabel("Correlation")

plt.show()


# ### Define Features and Target

# In[15]:


# Independent variables

X = df.drop(
    ['Class','ID'],
    axis=1
)


# Target variable

y = df['Class']


print("Feature Shape:", X.shape)
print("Target Shape:", y.shape)


# ### Train-Test Split

# In[16]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print(
    "Training Data:",
    X_train.shape
)

print(
    "Testing Data:",
    X_test.shape
)


# ### Feature Scaling

# In[17]:


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)


print("Feature scaling completed")


# ### Handle Class Imbalance Using SMOTE

# In[18]:


smote = SMOTE(
    random_state=42
)


X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_scaled,
    y_train
)


print(
    "Before SMOTE:",
    y_train.value_counts()
)


print(
    "After SMOTE:",
    y_train_balanced.value_counts()
)


# ### Create Model Evaluation Function

# In[19]:


def evaluate_model(model_name, model, X_test, y_test):

    prediction = model.predict(X_test)

    probability = model.predict_proba(X_test)[:,1]


    accuracy = accuracy_score(
        y_test,
        prediction
    )

    precision = precision_score(
        y_test,
        prediction
    )

    recall = recall_score(
        y_test,
        prediction
    )

    f1 = f1_score(
        y_test,
        prediction
    )

    auc = roc_auc_score(
        y_test,
        probability
    )


    results = pd.DataFrame({
        "Model":[model_name],
        "Accuracy":[accuracy],
        "Precision":[precision],
        "Recall":[recall],
        "F1 Score":[f1],
        "ROC-AUC":[auc]
    })


    return results


# ### Logistic Regression

# In[20]:


lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


lr_model.fit(
    X_train_balanced,
    y_train_balanced
)


lr_results = evaluate_model(
    "Logistic Regression",
    lr_model,
    X_test_scaled,
    y_test
)


lr_results


# ### Decision Tree

# In[21]:


dt_model = DecisionTreeClassifier(
    random_state=42
)


dt_model.fit(
    X_train_balanced,
    y_train_balanced
)


dt_results = evaluate_model(
    "Decision Tree",
    dt_model,
    X_test_scaled,
    y_test
)


dt_results


# ### Random Forest Baseline

# In[22]:


rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


rf_model.fit(
    X_train_balanced,
    y_train_balanced
)


rf_results = evaluate_model(
    "Random Forest",
    rf_model,
    X_test_scaled,
    y_test
)


rf_results


# ### Gradient Boosting

# In[23]:


gb_model = GradientBoostingClassifier(
    random_state=42
)


gb_model.fit(
    X_train_balanced,
    y_train_balanced
)


gb_results = evaluate_model(
    "Gradient Boosting",
    gb_model,
    X_test_scaled,
    y_test
)


gb_results


# ### XGBoost Model

# In[24]:


xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
)


xgb_model.fit(
    X_train_balanced,
    y_train_balanced
)


xgb_results = evaluate_model(
    "XGBoost",
    xgb_model,
    X_test_scaled,
    y_test
)


xgb_results


# ### Combine Model Results

# In[25]:


results = pd.concat(
    [
        lr_results,
        dt_results,
        rf_results,
        gb_results,
        xgb_results
    ],
    ignore_index=True
)


results


# ### Compare Models Graphically

# In[26]:


plt.figure(figsize=(10,5))

plt.bar(
    results['Model'],
    results['F1 Score']
)


plt.xlabel("Machine Learning Model")
plt.ylabel("F1 Score")

plt.title(
    "Comparison of Fraud Detection Models"
)

plt.xticks(rotation=45)

plt.show()


# ### Confusion Matrix for Best Model

# In[27]:


best_model = xgb_model


prediction = best_model.predict(
    X_test_scaled
)


cm = confusion_matrix(
    y_test,
    prediction
)


cm


# ### Display Classification Report

# In[28]:


print(
    classification_report(
        y_test,
        prediction
    )
)


# ### ROC Curve Comparison

# In[29]:


models = {
    "Logistic Regression":lr_model,
    "Decision Tree":dt_model,
    "Random Forest":rf_model,
    "Gradient Boosting":gb_model,
    "XGBoost":xgb_model
}


plt.figure(figsize=(8,6))


for name,model in models.items():

    probabilities = model.predict_proba(
        X_test_scaled
    )[:,1]


    fpr,tpr,threshold = roc_curve(
        y_test,
        probabilities
    )


    plt.plot(
        fpr,
        tpr,
        label=name
    )


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve Comparison"
)

plt.legend()

plt.show()


# ### Feature Importance of XGBoost

# In[30]:


importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":
    xgb_model.feature_importances_

})


importance = importance.sort_values(
    by="Importance",
    ascending=False
)


importance.head(10)


# ### Feature Importance Plot

# In[31]:


plt.figure(figsize=(10,6))

plt.bar(
    importance['Feature'].head(10),
    importance['Importance'].head(10)
)


plt.xlabel(
    "Features"
)

plt.ylabel(
    "Importance"
)


plt.title(
    "Top 10 Important Features Using XGBoost"
)


plt.xticks(rotation=45)

plt.show()


# ### Final Results

# In[32]:


results.to_csv(
    "fraud_detection_model_results.csv",
    index=False
)

print(
    "Results saved successfully"
)

