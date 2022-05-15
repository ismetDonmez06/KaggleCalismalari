"""
Yapacagımız adımlar

1-Gerekli kütüphanelerin indirilemsi
2-Bizden istenilen veriyi yüzeysel inceleme (data import)
3-Kayıp verileri düzeltcez
4-Verimizi detaylıca inceleme
5-Aykırı verilerin çıkaraılması
6-Feature Enginering yapacagız
7-Verimizi ayırcaz test -train
8-Verimizi standartlaşma yapacığız
9-Verimizi ML algoritmlarıyla eğitecez
10- En iyi paremetreleri bulup vsonuçları inceliyeceğiz

"""



import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor,LocalOutlierFactor,NeighborhoodComponentsAnalysis,KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import  neighbors
from sklearn.svm import SVR
from warnings import filterwarnings
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from scipy.stats import norm, skew, describe
import xgboost as xgb

filterwarnings("ignore")


data= pd.read_csv("bestsellers with categories.csv")

#print(data.columns)
#['Name', 'Author', 'User Rating', 'Reviews', 'Price', 'Year', 'Genre']
#print(data.head())
#print(data.shape)
#print(data.info())
#print(data.describe().T)

corr_matrix =data.corr()
"""sns.clustermap(corr_matrix,annot=True,fmt=".2f")
plt.show()"""

#Genre -Price türün fiyata etkisi
#Genre verimizi katerical yapıp örn erkese 1 kadınsa 0 gibi yaptım
data["Genre"]=data["Genre"].astype("category")
data["Genre"] = data["Genre"].cat.rename_categories({"Non Fiction":0, "Fiction":1})
#print(print(data["Genre"]))


#türün fiyata etkisi
"""plt.plot(data["Price"], marker = 'o')
plt.show()"""

print(data["Genre"].corr(data["Price"]))



#türün reytinge etkisi
#reyting gorseli
plt.plot(data["User Rating"], marker = 'o')
"""plt.show()"""

print(data["Genre"].corr(data["User Rating"]))


#hangi yıllarda hangi tür var
"""data1 =data.groupby(["Genre"]).sum()

plt.plot(data1,data["Year"], marker = 'o')
plt.show()"""


print(data.head())


#ismet dönmez olarak 2016 yılında kurgu bir film yazarak 10 dolardan satarsam reytingi ne olur


silincek=['Name', 'Author', 'Reviews',"User Rating"]

#print(data.info())
x=data.drop(silincek,axis=1)
y=data["User Rating"]

#aykırı değer bulma tekli değer için
q1=y.quantile(0.25)
q3=y.quantile(0.75)
iqr=q3-q1
alt_sinir = q1 - 1.5*iqr
ust_sinir= q3+1.5*iqr
print(alt_sinir,ust_sinir)
altsinir_atilacaklar =y<alt_sinir
ustsinir_atilacaklar =y>ust_sinir
"""print(y.shape)"""

aykiridegerler=y[((y < (alt_sinir)) | (y >(ust_sinir)))]
print(aykiridegerler)
print(y.mean())
mean=4.6
y[((y < (alt_sinir)) | (y >(ust_sinir)))] =mean
"""print(y[100:110])"""

"""y=y[((y > (alt_sinir)) & (y <(ust_sinir)))]
print(y.shape)"""


#Çok değişkenkli aykırı gözlem
#Local outlier Factor

"""clf =LocalOutlierFactor(n_neighbors=20, contamination=0.1)
clf.fit_predict(x)
df_scores=clf.negative_outlier_factor_

print(np.sort(df_scores)[0:20])

aykiri_deger_x=df_scores[10]

x=x[df_scores<aykiri_deger_x]
print(x)
"""



X_train,X_test ,Y_train,Y_test =train_test_split(x,y,test_size=0.9,random_state=42)




# lineer Regrosyonla veri eğitimi
"""lr = LinearRegression().fit(X_train,Y_train)
y_pred =lr.predict(X_test)
mse =mean_squared_error(Y_test,y_pred)
print(X_test)
print("Lineer Regresyon MSE :" ,mse)
isa=[[20,2016,1]]


y_tahmin = lr.predict(isa)
print()
print(y_tahmin)
"""