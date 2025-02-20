import pandas as pd
import numpy as np
import seaborn as sns
dataset=pd.read_csv(filepath_or_buffer="steam_cleaned.csv")
dataset.head()
#lets check for nulls
dataset.isnull().sum()
#lets remove all nulls
no_nulls=dataset.dropna()
no_nulls.isnull().sum()
no_nulls.head()
sns.pairplot(dataset,hue="Review_type")
#make the date format consistent
no_nulls["Release_date"]=pd.to_datetime(no_nulls["Release_date"],format="%Y-%m-%d")
#let's remove all free games 
zeros_index=[]
index_count=0
for _index_, value in no_nulls["Price"].items():
    if value==0:
        zeros_index.append(_index_)
valid_index=no_nulls["Price"].index.intersection(zeros_index)
no_zeros=no_nulls.drop(valid_index)
print(no_zeros.info())
no_zeros.head()
#lets remove outliers
price_mean=np.mean(no_zeros["Price"])
review_mean=np.mean(no_zeros["Review_no"])
price_std=np.std(no_zeros["Price"])
review_std=np.std(no_zeros["Review_no"])
outliers=[]
threshold=3
for price_index,price in no_zeros["Price"].items():
    z_score=np.abs(price-price_mean)/price_std
    if z_score>threshold:
        outliers.append(price_index)
for review_index,review in no_zeros["Review_no"].items():
    y_score=np.abs(review-review_mean)/review_std
    if y_score > threshold:
        outliers.append(review_index)
valid_outlier_index=no_zeros.index.intersection(outliers)
clean_data=no_zeros.drop(valid_outlier_index)
print(clean_data.info())
clean_data.head()
sns.pairplot(clean_data,hue="Review_type")
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
#lets split the input and output features
x_input=clean_data[["Price","Review_no"]]
y_input=clean_data["Review_type"]
x_scaler=StandardScaler()
x_scaled=x_scaler.fit_transform(x_input)
encoding=LabelEncoder()
y=encoding.fit_transform(y_input)
x_dataframe=pd.DataFrame(x_scaled,columns=["Price","Review_no"])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_dataframe,y, test_size=0.3)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_shape=(2,), activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(64, activation="sigmoid"))
model.add(Dense(128, activation="relu"))
model.add(Dense(7, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=100)
import numpy as np
predict = model.predict(x_test)
y_predict = [np.argmax(value) for value in predict]
y_true = [value for value in y_test[:10]]  

print(y_predict[0:10])
y_true
print("Y Prediction:  ",encoding.inverse_transform(y_predict[0:10]))
print("Y Real:        ",encoding.inverse_transform(y_true[0:10]))