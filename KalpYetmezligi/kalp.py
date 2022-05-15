import pandas as pd
import numpy as np
import sns
import seaborn as sns
import tips as tips
from pandas import get_dummies
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection, metrics
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


filterwarnings("ignore")

path ="heart_failure_clinical_records_dataset.csv"
data =pd.read_csv("heart_failure_clinical_records_dataset.csv")

#Verimizi yüzeysel inceleyelim

print(data.head())

print(data.shape)
print(data.info())
print(data.describe().T)

#Veriyi detaylıca inceleme
"""corr_matrix =data.corr()
sns.clustermap(corr_matrix,annot=True,fmt=".2f")
plt.show()

g = sns.countplot(data['DEATH_EVENT']) #checking imbalance after removing outliers
g.set_xticklabels(['0','1'])
plt.show()
"""


#Verilerin olum ornaına etkisi

"""dev =data.corrwith(data['DEATH_EVENT'])
print(dev)

sns.countplot(x='platelets',data=data)
plt.show()
"""
#outliner verileri çıkarma

cp_out=data["creatinine_phosphokinase"]
"""sns.boxplot(y=data["creatinine_phosphokinase"] )
plt.show()"""

q1_cp_out=data["creatinine_phosphokinase"].quantile(0.25)
q3_cp_out =data["creatinine_phosphokinase"].quantile(0.75)
igr_cp_out =q3_cp_out-q1_cp_out
low_lim_cp_out =q1_cp_out-1.5*igr_cp_out
up_lim_cp_out =q3_cp_out+1.5*igr_cp_out
out_cp_out=cp_out[((cp_out>(up_lim_cp_out)) | (cp_out<(low_lim_cp_out)))]
cp_out_mean=cp_out.mean()
cp_out[((cp_out>(up_lim_cp_out)) | (cp_out<(low_lim_cp_out)))]=cp_out_mean
print(data["creatinine_phosphokinase"])

sns.boxplot(y=data["creatinine_phosphokinase"] )
plt.show()
##########


sc_out=data["serum_creatinine"]
sns.boxplot(y=data["serum_creatinine"] )
plt.show()

q1_sc_out=data["serum_creatinine"].quantile(0.25)
q3_sc_out =data["serum_creatinine"].quantile(0.75)
igr_sc_out =q3_sc_out-q1_sc_out
low_lim_p_out =q1_sc_out-1.5*igr_sc_out
up_lim_p_out =q3_sc_out+1.5*igr_sc_out
out_sc_out=sc_out[((sc_out>(up_lim_p_out)) | (sc_out<(low_lim_p_out)))]
sc_out_mean=sc_out.mean()
sc_out[((sc_out>(up_lim_p_out)) | (sc_out<(low_lim_p_out)))]=sc_out_mean
print(data["serum_creatinine"])

sns.boxplot(y=data["serum_creatinine"] )
plt.show()
##################
ef_out =data["ejection_fraction"]


"""sns.boxplot(y=data["ejection_fraction"] )
plt.show()"""

q1_ef_out=data["ejection_fraction"].quantile(0.25)
q3_ef_out =data["ejection_fraction"].quantile(0.65)
igr_ef_out =q3_ef_out-q1_ef_out
low_lim_ef_out =q1_ef_out-1.5*igr_ef_out
up_lim_ef_out =q3_ef_out+1.5*igr_ef_out
out_ef_out=ef_out[((ef_out>(up_lim_ef_out)) | (ef_out<(low_lim_ef_out)))]
ef_out_mean=ef_out.mean()
ef_out[((ef_out>(up_lim_ef_out)) | (ef_out<(low_lim_ef_out)))]=ef_out_mean

print(data["ejection_fraction"])

sns.boxplot(y=data["ejection_fraction"] )
plt.show()








##################

y=data["DEATH_EVENT"]
x=data.drop("DEATH_EVENT",axis=1)


#train test -spilti



x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=1)



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
lr_ac = accuracy_score(y_test,lr_pred)
lr_con = confusion_matrix(y_test, lr_pred)
accuracies = []
accuracies.append(lr_ac)
print(lr_ac)
print(lr_con)
