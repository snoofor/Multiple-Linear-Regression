"""

Creating train and test dataset
Train/Test Split involves splitting the dataset into training and testing sets
that are mutually exclusive.
After which, you train with the training set and test with the testing set.
This will provide a more accurate evaluation on out-of-sample accuracy
because the testing dataset is not part of the dataset that have been used to train the model.
Therefore, it gives us a better understanding of how well our model generalizes on new data.

This means that we know the outcome of each data point in the testing dataset,
making it great to test with! Since this data has not been used to train the model,
the model has no knowledge of the outcome of these data points. So, in essence,
it is truly an out-of-sample testing.

Let's split our dataset into train and test sets.
80% of the entire dataset will be used for training and 20% for testing.
We create a mask to select random rows using np.random.rand() function:

"""

"""
path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
"""
###########################################################################

"""
Understanding the Data
FuelConsumption.csv:
We have downloaded a fuel consumption dataset, FuelConsumption.csv, 
which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions 
for new light-duty vehicles for retail sale in Canada. Dataset source

MODELYEAR e.g. 2014
MAKE e.g. Acura
MODEL e.g. ILX
VEHICLE CLASS e.g. SUV
ENGINE SIZE e.g. 4.7
CYLINDERS e.g 6
TRANSMISSION e.g. A6
FUEL CONSUMPTION in CITY(L/100 km) e.g. 9.9
FUEL CONSUMPTION in HWY (L/100 km) e.g. 8.9
FUEL CONSUMPTION COMB (L/100 km) e.g. 9.2
CO2 EMISSIONS (g/km) e.g. 182 --> low --> 0
"""

###########################################################################
# Imports

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from scipy import stats
import matplotlib as mpt
mpt.use('Qt5Agg')

# Setting
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

###########################################################################
# Reading Data

df_ = pd.read_csv("datasets/IBM/FuelConsumptionCo2.csv")
df = df_.copy()
df.columns

"""
Index(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'ENGINESIZE', 'CYLINDERS', 
'TRANSMISSION', 'FUELTYPE', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 
'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG', 'CO2EMISSIONS'], dtype='object')
"""

df.info()
###########################################################################
# pPrep Data

df.isnull().any() # No null value


def drop_object(dataframe):
    o_list = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    dataframe.drop(columns=o_list, axis=1, inplace=True)


drop_object(df)

df.head()

df.describe().T

df.columns

"""
Index(['MODELYEAR', 'ENGINESIZE', 'CYLINDERS', 
'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 
'FUELCONSUMPTION_COMB_MPG', 'CO2EMISSIONS'], dtype='object')
"""
###########################################################################

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

fig.suptitle('Independent and Dependent Variables Linear Relationship')

sns.regplot(ax=axes[0, 0], data=df, x='CO2EMISSIONS', y='MODELYEAR', color="blue", line_kws={"color": "r"})
sns.regplot(ax=axes[0, 1], data=df, x='CO2EMISSIONS', y='ENGINESIZE', color="blue", line_kws={"color": "r"})
sns.regplot(ax=axes[0, 2], data=df, x='CO2EMISSIONS', y='CYLINDERS', color="blue", line_kws={"color": "r"})
sns.regplot(ax=axes[1, 0], data=df, x='CO2EMISSIONS', y='FUELCONSUMPTION_CITY', color="blue", line_kws={"color": "r"})
sns.regplot(ax=axes[1, 1], data=df, x='CO2EMISSIONS', y='FUELCONSUMPTION_HWY', color="blue", line_kws={"color": "r"})
sns.regplot(ax=axes[1, 2], data=df, x='CO2EMISSIONS', y='FUELCONSUMPTION_COMB', color="blue", line_kws={"color": "r"})

df.drop("MODELYEAR", axis=1, inplace=True)

df.describe().T
###########################################################################

df_h = df.copy()
df_h.columns
"""
Index(['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 
'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG'], dtype='object')
"""
df_h.drop("CO2EMISSIONS", axis=1, inplace=True)

# don't forget there is no dependent variable
plt.figure(figsize=(8, 6))
plt.title("Independent variables correlation")
p=sns.heatmap(df_h.corr(), annot=True, cmap='inferno', square=True)

df_h.drop(["FUELCONSUMPTION_CITY", "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB_MPG"], axis=1, inplace=True)


from statsmodels.stats.outliers_influence import variance_inflation_factor

# the independent variables set
X = df_h[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]

print(vif_data)

###########################################################################
# Model Fit for Multiple Linear Regression

df.drop(["FUELCONSUMPTION_CITY", "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB_MPG"], axis=1, inplace=True)

df.columns
df.info()

"""
Index(['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS'], dtype='object')
"""

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

from sklearn import linear_model
regr = linear_model.LinearRegression()
x_train = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x_train, y_train)
# The coefficients
print('Coefficients: ', regr.coef_)

###########################################################################
# Prediction

y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x_test = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y_test) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x_test, y_test))

resid = y_test - y_hat

###########################################################################
# Get DataFrame both test and prediction values

test_y_list = pd.DataFrame(y_test.tolist())
test_predict_list = pd.DataFrame(y_hat.tolist())
result = pd.concat([test_y_list, test_predict_list], axis=1)
result.columns = ["CO2EMISSIONS", "Prediction"]
result["Error"] = result["CO2EMISSIONS"] - result["Prediction"]
result["Error-square"] = result["Error"]**2

result.head(15)

###########################################################################

residuals = result["Error"]
mean_residuals = residuals.mean()
print("Mean of Residuals {}".format(mean_residuals))

plt.title("Homoscedasticity Check")
sns.regplot(data=result, x='Prediction', y='Error', color="blue", line_kws={"color": "r"})
plt.show()

# The null hypothesis (H0): Homoscedasticity is present.
# The alternative hypothesis: (Ha): Homoscedasticity is not present (i.e. heteroscedasticity exists)

import statsmodels.formula.api as smf

#fit regression model
fit = smf.ols('CO2EMISSIONS ~ ENGINESIZE+CYLINDERS+FUELCONSUMPTION_COMB', data=test).fit()

#view model summary
print(fit.summary())

from statsmodels.compat import lzip
import statsmodels.stats.api as sms
name = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(fit.resid, fit.model.exog)
lzip(name, test)

# p_value is not less than 0.05 so, H0 can not be denied.

###########################################################################


# The null hypothesis (H0): Observations are independent.
# The alternative hypothesis: (Ha): Observations are not independent

# durbin watson
from statsmodels.stats.stattools import durbin_watson
resid
diff_resids = np.diff(resid, 1, axis=0)
dw = np.sum(diff_resids**2, axis=0) / np.sum(resid**2, axis=0)
dw
# dw 1.892


###################################
# Residuals are normally distributed

import pylab
import scipy.stats as stats

stats.probplot(result["Error"], dist="norm", plot=pylab)
pylab.show()

###########################################################################
# Final Results

result.head(15)