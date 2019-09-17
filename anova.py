#!/usr/bin/env python
# coding: utf-8

# In[322]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import researchpy as rp
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from datetime import datetime


# In[323]:


def huan1(z):
    '''
    a is the window.
    b is the center of function.
    z is the height or altitude in this case.(numpy array or one value)
    '''

    logical_positive = np.logical_and(z != np.nan, z > 0)
    logical_negative = np.logical_and(z != np.nan, z < 0)
    tmpvalue = np.where(logical_positive, 1, np.where(logical_negative, -1,
                                                      0))
    return tmpvalue


def huan(z):
    '''
    a is the window.
    b is the center of function.
    z is the height or altitude in this case.(numpy array or one value)
    '''
    logical_positive = np.logical_and(z != np.nan, z > 0)

    tmpvalue = np.where(logical_positive, 1, -1)
    return tmpvalue


# 载入数据Loading data
df1 = pd.read_csv("berra.csv")
df2 = pd.read_csv("ZX3.csv")
df3 = pd.read_csv("hg.csv")
print(df2.dtypes)
# df1["date"]=df1["date"].astype('datetime64')
# df2["date"]=df2["date"].astype('datetime64')
df3["date"] = df3["date"].astype('datetime64')
df1["date"] = pd.to_datetime(df1["date"])
df2["date"] = pd.to_datetime(df2["date"])
df1.set_index("date", inplace=True)
df2.set_index("date", inplace=True)
df2 = df2.convert_objects(convert_numeric=True)
print(df2.dtypes)
# 按时间合并两个表 2010-01-04---2019-06-03
index = pd.merge(df1, df2, on='date')

index = (index.shift(-1) - index) / index
for col in index:
    index[col] = huan1(index[col])
print(index)
hg = df3.copy()
hg.set_index("date", inplace=True)
# print(hg)
# df = pd.merge(df, df3, on='date')


# In[324]:


print(index.dtypes)

# In[325]:


# Recoding value from numeric to string
# test=pd.DataFrame(np.arange(float(20)).reshape(4,5))
# test.iloc[1,3]=np.nan
# print(test)
# tt=test.rolling(2,min_periods=2,axis=0).mean()
# print(hg)


hg_rmean = hg.rolling(4, min_periods=2, axis=0).mean()
hg_rmean = hg_rmean['2010-01-04':'2019-06-03']
hg_ratio = hg_rmean.copy()

# hg_ratio=hg_ratio.replace(np.nan,0)

# print(hg_ratio)
hg_ratio['dq'] = (hg_ratio['dq'] - hg_ratio['dq'].shift(1)) / hg_ratio['dq'].shift(1)
hg_ratio['zcq'] = (hg_ratio['zcq'] - hg_ratio['zcq'].shift(1)) / hg_ratio['zcq'].shift(1)
hg_ratio['gjj'] = (hg_ratio['gjj'] - hg_ratio['gjj'].shift(1)) / hg_ratio['gjj'].shift(1)
hg_ratio['ding3'] = (hg_ratio['ding3'] - hg_ratio['ding3'].shift(1)) / hg_ratio['ding3'].shift(1)
hg_ratio['ding6'] = (hg_ratio['ding6'] - hg_ratio['ding6'].shift(1)) / hg_ratio['ding6'].shift(1)
hg_ratio['zyxm'] = (hg_ratio['zyxm'] - hg_ratio['zyxm'].shift(1)) / hg_ratio['zyxm'].shift(1)
hg_ratio['dfxm'] = (hg_ratio['dfxm'] - hg_ratio['dfxm'].shift(1)) / hg_ratio['dfxm'].shift(1)
# hg_ratio=hg_ratio.replace(np.nan,0)

for col in hg_ratio:
    hg_ratio[col] = huan(hg_ratio[col])
print(hg_ratio)
# hg_r=hg.rolling()
# hg['dq'].replace({1: 'placebo', 2: 'low', 3: 'high'}, inplace= True)
#
# # Gettin summary statistics
# rp.summary_cont(df['libido'])
# 'zcq','gjj','ding3', 'ding6','zyxm','dfxm'
# rp.summary_cont(df['libido'].groupby(df['dose']))


# In[326]:


# hg_ratio.cumsum()
# hg_ratio.plot()
# plt.show()


# In[327]:


# print(rp.summary_cont(index['beta']))
# print(rp.summary_cont(index['size']))
# print(rp.summary_cont(index['leverage']))
# print(rp.summary_cont(index['book_to_price']))
# print(rp.summary_cont(index['earning_yield']))
# print(rp.summary_cont(index['growth']))
# print(rp.summary_cont(index['liquidity']))
# print(rp.summary_cont(index['nl_size']))
# print(rp.summary_cont(index['momentum']))
# print(rp.summary_cont(index['volatility']))
# print(rp.summary_cont(index['HuoD']))
# print(rp.summary_cont(index['FaD']))
# print(rp.summary_cont(index['XiD']))


# In[328]:


mm = pd.concat([index, hg_ratio], axis=1)
print(mm)
# print(index)
# hg_ratio = np.where(hg_ratio[""] < 0, "N", "P")
# print(hg_ratio)


# In[329]:


# index.cumsum()
# index.plot()
# plt.show()
#
# hg_ratio.cumsum()
# hg_ratio.plot()
# plt.show()


# In[330]:


# hg_ratio[["dq","zcq","gjj","ding3","ding6"]] = np.where(hg_ratio[["dq","zcq","gjj","ding3","ding6"]]< 0, "N", "P")
# print(hg_ratio)
# hg_cat=hg_ratio.copy()


#
# hg_cat["dq"]=huan(hg_ratio["dq"])
# hg_cat["zcq"]=huan(hg_ratio["zcq"])
# hg_cat["gjj"]=huan(hg_ratio["gjj"])
# hg_cat["ding3"]=huan(hg_ratio["ding3"])
# hg_cat["ding6"]=huan(hg_ratio["ding6"])
# hg_cat["zyxm"]=huan(hg_ratio["zyxm"])
# hg_cat["dfxm"]=huan(hg_ratio["dfxm"])

# mm = pd.concat([index,hg_cat],axis=1)
# mm.resample('M').dropna()
# pd.DataFrame(mm.index).to_period(freq='M',axis=0,copy=False)
# mm.index=pd.dataFrame.to_string(mm["date"])
d = []

for i in mm.index:
    i = i.strftime("%Y-%m")
    d.append(i)

# In[331]:


# print(d)
mm.index = d
# print(mm)
# print(date)
# d_rows=pd.DataFrame.duplicated(mm,keep=False)
d_rows = mm[mm.index.duplicated(keep=False)]
# print(d_rows)
print(mm)

# In[332]:


mm = mm.groupby(mm.index).sum().dropna()
print(mm)
# print(rp.summary_cont(mm['growth'].groupby(mm['dq'])))


# In[333]:


# stats.f_oneway(mm['growth'][mm['dq'] == 'pos'],
#                mm['growth'][mm['dq'] == 'neg'],
#                mm['growth'][mm['dq'] == 'no_change'])


# In[334]:


results = ols('FaD ~ C(ding6)', data=mm).fit()
# print(results.summary())

aov_table = sm.stats.anova_lm(results, typ=2)
print(aov_table)

# In[335]:


results = ols('growth ~ C(dq)', data=mm).fit()
print(results.summary())

aov_table = sm.stats.anova_lm(results, typ=2)
print(aov_table)

# In[311]:


results = ols('XiD ~ C(dfxm)', data=mm).fit()
print(results.summary())

aov_table = sm.stats.anova_lm(results, typ=2)
print(aov_table)

# In[258]:


# # datet=mm.index.copy()
# #
# import datetime as dt
# # datetime2 = [(x.shiftime('%Y-%m')) for x in datet]
# mm.index.strftime('%Y-%m')
# # mm["date"]= datetime2
# #
#
# mm.groupby(mm.index).ffill().groupby(mm.index).last()
# mm.index = mm.index.droplevel(0)
# print(mm)


# In[259]:


# In[229]:


# In[ ]:




