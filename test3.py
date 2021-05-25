

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing as fch  #加载加利福尼亚房屋价值数据
#加载线性回归需要的模块和库
import statsmodels.api as sm #最小二乘
from statsmodels.formula.api import ols #加载ols模型


#设置全部行输出
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


data = fch() #导入数据
house_data=pd.DataFrame(data.data) #将自变量转换成dataframe格式，便于查看
house_data.columns=data.feature_names  #命名自变量
house_data.loc[:,"value"]=data.target #合并自变量，因变量数据
print(house_data.shape) #查看数据量
print(house_data.head(10)) #查看前10行数据

#分训练集测试集
import random
random.seed(123) #设立随机数种子
a=random.sample(range(len(house_data)),round(len(house_data)*0.3))
house_test=[]
for i in a:
    house_test.append(house_data.iloc[i])
house_test=pd.DataFrame(house_test)
house_train=house_data.drop(a)

#重新排列index
for i in [house_test,house_train]:
    i.index = range(i.shape[0])
house_test.head()
house_train.head()

#训练模型
lm=ols('value~ MedInc + HouseAge + AveRooms + AveBedrms + Population + AveOccup + Latitude + Longitude',data=house_train).fit()

lm.summary()

#利用测试集测试模型
house_test.loc[:,"pread"]=lm.predict(house_test)
#计算R方
##计算残差平方和
error2=[]
for i in range(len(house_test)):
    error2.append((house_test.pread[i]-house_test.loc[:,"value"][i])**2)
##计算总离差平方和
sst=[]
for i in range(len(house_test)):
    sst.append((house_test.value[i]-np.mean(house_test.value))**2)
R2=1-np.sum(error2)/np.sum(sst)
print("R方为:",R2)


#作预测效果图

import matplotlib.pyplot as plt
plt.plot(range(len(house_test.pread)),sorted(house_test.value),c="black",label= "target_data")
plt.plot(range(len(house_test.pread)),sorted(house_test.pread),c="red",label = "Predict")
plt.legend()
plt.show()
