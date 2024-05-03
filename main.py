import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor 
# read a dataset
df=pd.read_csv ("FINAL_USO.csv")
df=df[["Date","SP_open","SP_high","SP_low","SP_close"]]
print(df.info()) 

# Spliting Date columns 
df["Date"]= pd.to_datetime(df["Date"])
df["year"]=df["Date"].dt.year 
df["day"]=df["Date"].dt.day 
df["month"]=df["Date"].dt.month

# For checking is there any null values 
print(df.isna().sum())

# for Detecting the outliers 
for column in ["SP_open","SP_high","SP_low"]:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    filter = (df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)
    df = df[filter]

print(df.info())


# determining is it left skewed or right for (preproccing)
df.hist()
plt.show()


# Spliting the train columns x and test column  y.
y=df["SP_close"]
x=df[["SP_open","SP_high","SP_low","day","year","month"]]

print(x.head())
print(y.head())

#  Spliting the train columns x and test column  y.

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1 , test_size=0.2 )

# create instance from LinearRegresiion 
linreg= LinearRegression()
linreg.fit(x_train,y_train)

# Prediction 

predict= linreg.predict(x_test) 
print(predict.shape)
sns.scatterplot(x=y_test,y=predict) 
sns.lineplot(x=y_test,y=y_test) # Actual values 
plt.show()

print(linreg.score(x_test,y_test)) # print the score of linear reg 


randreg=RandomForestRegressor() 
randreg.fit(x_train,y_train)

print(randreg.score(x_test,y_test))
