# AI USAGE ASSIGNMENT
## Part A: Basic EDA(1-8)
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

### 1.Load the dataset and display the first 5 rows
#load dataset
df =  pd.read_csv('ai_assistant_usage_student_life.csv')
df.head()# displayng the first 5 rows of the dataset
### 2.Check the dataset shape (rows, columns).
#check dataset shape (rows, columns)
print(df.shape)
### 3.Display column names and their data types.

#check column names and their data types(summary)
print(df.columns)
print(df.info())
### 4.Check for missing values in each column.
 
#checking missing values
df.isnull().sum()
- there are no missing values in the dataset.
### 5.Show summary statistics for SessionLengthMin and TotalPrompts.
#summary statistics for SessionLengthMin and TotalPrompts
print(df[['SessionLengthMin','TotalPrompts']].describe())
### 6.Find the number of unique values in StudentLevel, Discipline, and TaskType.
#find unique values in studentlevel,discipline and tasktype
print(df['StudentLevel'].unique())
print(df['Discipline'].unique())
print(df['TaskType'].unique())

### 7.Which TaskType is the most common?
#which tasktype is common
common_task=df['TaskType'].value_counts()
print(common_task)
- the common task was writing .
- you can put idxmax() at the end to list the task with most id assigned to it.
### 8.Calculate the average SessionLengthMin for each StudentLevel.
#average sessionlengthmin for each studentlevel
avg_session_by_level=df.groupby('StudentLevel')['SessionLengthMin'].mean()
print(avg_session_by_level)
## Part B: VISUALIZATION(9-16)
### 9.Plot a histogram of SessionLengthMin.
#plot histogramn of SessionLengthMin
plt.figure(figsize=(10,6))
plt.hist(df['SessionLengthMin'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Session Length (Minutes)')
plt.xlabel('Session Length (Minutes)')
plt.ylabel('Frequency')
plt.show()
### 10.Create a bar chart of session counts by StudentLevel.

#create barchart of Session Counts by StudentLevel
plt.figure(figsize=(10, 6))
df['StudentLevel'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Number of Sessions by Student Level')
plt.xlabel('Student Level')
plt.ylabel('Number of Sessions')
plt.xticks(rotation=45)
plt.show()
### 11.Make a countplot of TaskType using Seaborn.
#make a countplot of TaskType using Seaborn
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='TaskType', order=df['TaskType'].value_counts().index)
plt.title('Count of Sessions by Task Type')
plt.xlabel('Task Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
### 12.Plot a boxplot of SessionLengthMin grouped by StudentLevel.
# Plot a boxplot of SessionLengthMin grouped by StudentLevel
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='StudentLevel', y='SessionLengthMin')
plt.title('Session Length by Student Level')
plt.xlabel('Student Level')
plt.ylabel('Session Length (Minutes)')
plt.xticks(rotation=45)
plt.show()

### 13.Create a pie chart showing proportions of FinalOutcome.
# Create a pie chart showing proportions of FinalOutcome
plt.figure(figsize=(8, 8))
df['FinalOutcome'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Proportion of Final Outcomes')
plt.ylabel('')
plt.show()
### 14.Draw a scatterplot of SessionLengthMin vs. TotalPrompts.

# Draw a scatterplot of SessionLengthMin vs. TotalPrompts
plt.figure(figsize=(10, 6))
plt.scatter(df['SessionLengthMin'], df['TotalPrompts'], alpha=0.5)
plt.title('Session Length vs. Total Prompts')
plt.xlabel('Session Length (Minutes)')
plt.ylabel('Total Prompts')
plt.show()
# Convert SessionDate to datetime for time-based analysis
df['SessionDate'] = pd.to_datetime(df['SessionDate'])
### 15.Plot a line chart of average AI_AssistanceLevel over time (SessionDate).
# Plot a line chart of average AI_AssistanceLevel over time (SessionDate)
plt.figure(figsize=(12, 6))
df.groupby('SessionDate')['AI_AssistanceLevel'].mean().plot()
plt.title('Average AI Assistance Level Over Time')
plt.xlabel('Date')
plt.ylabel('Average AI Assistance Level')
plt.show()
### 16.Create a heatmap of correlations among numeric features.
# Create a heatmap of correlations among numeric features
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numeric Features')
plt.show()
## Part C: GROUPBY & AGGREGATIONS(17-21)

### 17.Find the average SessionLengthMin for each TaskType.
# calculating average sessionlengthmin for each tasktype
avg_session_by_task=df.groupby('TaskType')['SessionLengthMin'].mean()
print(avg_session_by_task)
### 18.Which Discipline had the most sessions?
#which discipline had the most sessions?
discipline_high=df["Discipline"].value_counts()
print(discipline_high)
- discipline with the highest sessions is Biology with 1458 sessions.
### 19.Compare average AI_AssistanceLevel across StudentLevel
avg_ai_by_level = df.groupby('StudentLevel')['AI_AssistanceLevel'].mean()
print(avg_ai_by_level)
### 20.Find the most common FinalOutcome for Graduate students.

#find the most common finaloutcome  for graduate students
grad_df = df[df['StudentLevel']=='Graduate']
most_outcome_grad = grad_df['FinalOutcome'].value_counts()
print(most_outcome_grad)
### 21.Calculate the median SessionLengthMin for each FinalOutcome.
#calculate median sessionlengthmin for finaloutcome
median_session_outcome = df.groupby('FinalOutcome')['SessionLengthMin'].median()
print(median_session_outcome)
## Part D:FEATURE ENGINEERING AND ENCODING(22-26)
### 22.Convert SessionDate into Year, Month, and Day columns.
#convert sessiondate into year, month, and day columns
df['Year'] = df['SessionDate'].dt.year
df['Month'] = df['SessionDate'].dt.month
df['Day'] = df['SessionDate'].dt.day

print("Dataset with new date columns")# heading for the dataset
print(df[['SessionDate','Year','Month','Day']].head())
### 23.Encode StudentLevel using Label Encoding.
#label encoding used to convert categorical data into a numerical format suitable for machine learning models. 
#encode StudentLevel using label encoding
le = LabelEncoder()
df['StudentLevel_Encoded'] = le.fit_transform(df['StudentLevel'])
print(df[['StudentLevel','StudentLevel_Encoded']].head(10))
### 24.Apply One-Hot Encoding to TaskType.
#to give zero and ones instead of false and true
df_encoded = pd.get_dummies(df['TaskType'],drop_first=True)
df_encoded= df_encoded.astype(int)
print(df_encoded.head())
- alternative if you want false or true instead of one or zeros
#apply one-hot encoding to tasktype
from encodings.uu_codec import uu_encode

task_type_dummies = pd.get_dummies(df['TaskType'],prefix='Task')
df = pd.concat([df,task_type_dummies],axis = 1)
print(df.filter(like = 'Task').head())

- One Hot Encoding is a method for converting categorical variables into a binary format. It creates new columns for each category where 1 means the category is present and 0 means it is not. The primary purpose of One Hot Encoding is to ensure that categorical data can be effectively used in machine learning models.
### 25.Create a new feature: PromptsPerMinute = TotalPrompts / SessionLengthMin.

#create a  new feature :promptsperminute = totalprompts/sessionlengthmin
df['PromptsPerMinute'] = df['TotalPrompts']/df['SessionLengthMin']
print(df[['TotalPrompts','SessionLengthMin','PromptsPerMinute']].head())
### 26.Bin SessionLengthMin into categories: Short, Medium, Long.
#bin sessionlengthmin into categories:Short , Medium, Long
bins = [0,10,30,df['SessionLengthMin'].max()]
labels = ['Short','Medium','Long']
df['SessionLengthCategory'] = pd.cut(df['SessionLengthMin'],bins = bins, labels=labels)
print(df['SessionLengthCategory'].value_counts())
## Part E: Machine Learning (Classification Models)(27-36)
#importing necessary libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score,recall_score,confusion_matrix, classification_report
import xgboost as xgb


# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
# Convert categorical variables to numerical(ENCODING)
le = LabelEncoder()
df['StudentLevel_encoded'] = le.fit_transform(df['StudentLevel'])
df['Discipline_Encoded'] = le.fit_transform(df['Discipline'])
df['TaskType_Encoded'] = le.fit_transform(df['TaskType'])
df['FinalOutcome_encoded'] = le.fit_transform(df['FinalOutcome'])
df['UsedAgain_encoded'] = le.fit_transform(df['UsedAgain'])
# Select features for models
features = ['StudentLevel_encoded', 'Discipline_Encoded', 'SessionLengthMin', 
            'TotalPrompts', 'TaskType_Encoded', 'AI_AssistanceLevel', 'SatisfactionRating']

#split features and targets
X = df[features]
y_outcome = df['FinalOutcome_encoded']

### 34.Split the dataset into 80% training and 20% testing sets.
X_train, X_test, y_outcome_train, y_outcome_test = train_test_split(X, y_outcome, test_size=0.2, random_state=42)



# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data preparation complete. Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
### 27.Predict FinalOutcome using a Decision Tree Classifier.
# 1. Predict FinalOutcome using a Decision Tree Classifier
dt_outcome = DecisionTreeClassifier(random_state=42)
dt_outcome.fit(X_train, y_outcome_train)
y_pred_dt_outcome = dt_outcome.predict(X_test)
accuracy_dt_outcome = accuracy_score(y_outcome_test, y_pred_dt_outcome)
print(f"Decision Tree Accuracy for FinalOutcome: {accuracy_dt_outcome:.4f}")
# 2. Evaluation(confusion marix)
print("Accuracy:", accuracy_score(y_outcome_test, y_pred_dt_outcome))
print("Precision (macro):", precision_score(y_outcome_test, y_pred_dt_outcome, average='macro'))
print("\nConfusion Matrix:\n", confusion_matrix(y_outcome_test, y_pred_dt_outcome))

#print("\nClassification Report:\n", classification_report(y_outcome_test, y_pred_dt_outcome, labels=labels, target_names=le.inverse_transform(labels)))
#print("\nDetailed Classification Report:\n", classification_report(y_outcome_test, y_pred_dt_outcome, target_names=le.classes_))
### 28.Predict UsedAgain using Logistic Regression.
# Define features (same as before)
tasktype_features = [col for col in df.columns if col.startswith("TaskType_")]

features = [
    'SessionLengthMin',
    'TotalPrompts',
    'AI_AssistanceLevel',
    'PromptsPerMinute',
    'StudentLevel_encoded',
    'Discipline_encoded',
    'Year', 'Month', 'Day'
] + tasktype_features

X = df[features]
y = df["UsedAgain_encoded"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Train
log_reg.fit(X_train, y_train)

# Predict
y_pred_log = log_reg.predict(X_test)
#logistic regression accuracy
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print(f"Logistic Regression Accuracy for UsedAgain: {accuracy_log_reg:.4f}")
#Evaluate(Confusion matrix, classification report)
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision (macro):", precision_score(y_test, y_pred_log, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred_log, average='macro'))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("\nDetailed Classification Report:\n",classification_report(y_test, y_pred_log, target_names=[str(c) for c in log_reg.classes_]))
### 29.Train a Random Forest Classifier to predict FinalOutcome.
model_rf= RandomForestClassifier(n_estimators=100,random_state=42)#initialize model
model_rf.fit(X_train,y_train)#train model
y_pred_rf=model_rf.predict(X_test)#predict model

accuracy_rf_outcome = accuracy_score(y_test,y_pred_rf)#test accuracy
print(f"Random Forest Accuracy for FinalOutcome: {accuracy_rf_outcome:.4f}")

# Evaluation(Accuracy, Precision, Confusion matrix)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision (macro):", precision_score(y_test, y_pred_rf, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred_rf, average='macro'))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

#labels = np.unique(y_test)  # only use labels present in y_test
#print("\nClassification Report:\n", classification_report(y_test, y_pred_rf, labels=labels, target_names=le.inverse_transform(labels)))

### 30.Use KNN (K-Nearest Neighbors) to classify UsedAgain.
#initialize model
knn=KNeighborsClassifier(n_neighbors=5)#you can tune the n_neighbors 
knn.fit(X_train,y_train)#train model
y_pred_knn=knn.predict(X_test)#predict model
accuracy_score_knn=accuracy_score(y_test,y_pred_knn)#predict accuracy
print(f"KNN Accuracy score is:{accuracy_score_knn:.4f}")
# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Precision (macro):", precision_score(y_test, y_pred_knn, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred_knn, average='macro'))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
#print("\nDetailed Classification Report:\n",classification_report(y_test, y_pred_knn, target_names=[str(c) for c in le_used.classes_]))
### 31.Train a Naive Bayes Classifier to predict FinalOutcome.
model_nb=GaussianNB()#initialize the model
model_nb.fit(X_train,y_train)#train the model
y_pred_nb = model_nb.predict(X_test)#predict the model
accuracy_score_nb=accuracy_score(y_test,y_pred_nb)#predict accuracy score
print(f"Naive Bayes accuracy is:{accuracy_score_nb:.4f}")
#confusion matrix ,precision,accuracy scores
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Precision (macro):", precision_score(y_test, y_pred_nb, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred_nb, average='macro'))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))

### 32.Build a Gradient Boosting Classifier for UsedAgain.
gb_classifier= GradientBoostingClassifier(random_state=42)#initialize
gb_classifier.fit(X_train,y_train)#train model
y_pred_gb =gb_classifier.predict(X_test)#predict model
accuracy_score_gb=accuracy_score(y_test,y_pred_gb)#test model accuracy
print (f"Gradient Boosting Classifier :{accuracy_score_gb:.4f}")
#confusion matrix,precision , accuracy
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("Precision (macro):", precision_score(y_test, y_pred_gb, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred_gb, average='macro'))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_gb))
#print("\nClassification Report:\n", classification_report(y_test, y_pred_gb, target_names=[str(c) for c in le_used.classes_]))
### 33.Apply an XGBoost Classifier to predict FinalOutcome.
xgb=XGBClassifier(use_label_encoder=False,eval_metric='mlogloss',random_state=42)
xgb.fit(X_train,y_train)
y_pred_xgb=xgb.predict(X_test)
accuracy_score_xgb=accuracy_score(y_test,y_pred_xgb)
print(f"accuracy xgb score for finaloutcome is:{accuracy_score_xgb:.4f}")
#confusion matrix, accuracy, precision
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Precision (macro):", precision_score(y_test, y_pred_xgb, average='macro'))
print("Recall (macro):", recall_score(y_test, y_pred_xgb, average='macro'))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
#print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb, target_names=le.classes_))
### 35.Evaluate one model using accuracy and a confusion matrix.
#evaluate logistic regression model
print("\nLogistic Regression Evaluation for UsedAgain:")
print(f"Accuracy: {accuracy_log_reg:.4f}")

#confusion matrix plot
cm_lr = confusion_matrix(y_test,y_pred_log)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression (UsedAgain)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
### 36.Generate a classification report (precision, recall, f1-score).
print("\nClassification Report for Logistic Regression (UsedAgain):")
print(classification_report(y_test, y_pred_log, 
                            target_names=['Not Used Again', 'Used Again']))
## Part F:Model Evaluation & Hyperparameter Tuning(37-40)
### 37.Perform cross-validation for Logistic Regression.
cv_scores_lr = cross_val_score(log_reg, X_train_scaled, y_train, cv=5)
print(f"Logistic Regression Cross-Validation Scores: {cv_scores_lr}")
print(f"Mean CV Accuracy: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std() * 2:.4f})")
### 38.Use GridSearchCV to tune hyperparameters of a Decision Tree.
param_grid_dt = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt = DecisionTreeClassifier(random_state=42)
grid_search_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring='accuracy')
grid_search_dt.fit(X_train, y_outcome_train)

print("\nBest parameters for Decision Tree:")
print(grid_search_dt.best_params_)
print(f"Best cross-validation score: {grid_search_dt.best_score_:.4f}")
### 39.Tune a Random Forest Classifier (n_estimators, max_depth).

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train, y_outcome_train)

print("\nBest parameters for Random Forest:")
print(grid_search_rf.best_params_)
print(f"Best cross-validation score: {grid_search_rf.best_score_:.4f}")
### 40.Compare Logistic Regression, Decision Tree, Random Forest, Naive Bayes, KNN, Gradient Boosting, and XGBoost for predicting UsedAgain.

