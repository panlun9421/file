import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import KNNImputer

file_path1 = "D:\\filedownload\siri\\train_data .csv"
file_path2 = "D:\\filedownload\siri\\test_data.csv"

train_data = pd.read_csv(file_path1)
test_data = pd.read_csv(file_path2)

train_label = train_data['label']
train_feature = train_data.drop(columns=['label'])
test_label = test_data['label']
test_feature = test_data.drop(columns=['label'])

imputer = KNNImputer(n_neighbors=2)
train_feature_fit = imputer.fit_transform(train_feature)
test_feature_fit = imputer.fit_transform(test_feature)

scaler = StandardScaler()
train_feature = scaler.fit_transform(train_feature)
test_feature = scaler.fit_transform(test_feature)

clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(train_feature_fit,train_label)

y_pred = clf.predict(test_feature_fit)
accuracy = classification_report(y_pred,test_label)
print(accuracy)
