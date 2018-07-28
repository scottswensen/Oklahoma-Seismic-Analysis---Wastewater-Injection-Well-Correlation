"""The purpose of this machine learning script is to investigate the link
between oil and gas industry injection wells in Oklahoma and earthquake
activity in the area and determine if the number of future earthquakes near an 
active injection well may be predicted based on well location, injection depth,
underlying rock formation, and injection pressures and volumes. The Oklahoma 
Corporation Comission (OCC) requires that injection well owners submit data 
giving the monthly injection volume and pressure for all wastewater disposal 
and enhanced revoery wells in the state. That data is publicly available from 
2011 to 2017 (http://www.occeweb.com/og/ogdatafiles2.htm). The number and 
location of earthquakes of Magnitude 2.5 or greater recorded in Oklahoma were 
also obtained through the United States Geological Survey for 2017 (USGS) at 
https://earthquake.usgs.gov/earthquakes/search.
"""
# Import libraries and analysis tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#
from mpl_toolkits.basemap import Basemap
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
#
# Open cleaned data file with well injection data
df = pd.read_csv('full_data.csv')
# Open data file with earthquakes of M >= 2.5 in 2017
df_EQ = pd.read_csv('OK_EQ.csv')
df['Vol_Tot'] = df[['Vol_2017','Vol_2016','Vol_2015','Vol_2014','Vol_2013','Vol_2012','Vol_2011']].sum(axis = 1)
df['vol_wells5'] = df[['vol_wells5_2017','vol_wells5_2016','vol_wells5_2015','vol_wells5_2014','vol_wells5_2013','vol_wells5_2012','vol_wells5_2011']].sum(axis = 1)
df['vol_wells10'] = df[['vol_wells10_2017','vol_wells10_2016','vol_wells10_2015','vol_wells10_2014','vol_wells10_2013','vol_wells10_2012','vol_wells10_2011']].sum(axis = 1)
#
#PLOT SOME OF THE DATA
#
# Draw a scatter plot assigning point sizes to injection volumes and color to 
# mean pressure
sns.set(style="whitegrid")
#
# Draw the map background
fig = plt.figure(figsize=(12, 5))
m = Basemap(projection='lcc', resolution='h', 
            lat_0=35.5, lon_0=-98.6,
            width=8.5E5, height=5.0E5)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')
#
# Scatter plot of injection data, with color reflecting well type and size 
# reflecting 2017 injection volume
m.scatter(list(df['Lon']), list(df['Lat']), latlon=True, 
          s = list(df['Vol_Tot']/250000), alpha = 0.5, 
          c = list(df['Pres_2017']), cmap='Reds')
#
# Create colorbar and legend
plt.colorbar(label=r'$Mean Pressure({\rm PSI})$')
plt.clim(0, 5000)
plt.title('Oklahoma Injection Well Volume (2011-2017) and Mean Pressure (2017)', fontsize = 14)
#
# Make legend with dummy points
for a in [1000000/250000, 20000000/250000, 40000000/250000, 60000000/250000, 80000000/250000, 100000000/250000]:
    plt.scatter([], [], c='k', alpha=0.5, s=a,
                label=str(np.round(a*250000/1000000)) + ' million barrels')
plt.legend(scatterpoints=1, frameon=False,
           labelspacing=1, loc='lower left')
plt.show()
#
# Draw a scatter plot showing earthquake locations
#
# Draw the map background
fig = plt.figure(figsize=(12, 5))
m = Basemap(projection='lcc', resolution='h', 
            lat_0=35.5, lon_0=-98.6,
            width=8.5E5, height=5.0E5)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')
#
# scatter earthquake data, with color reflecting well type
# and size reflecting 2017 injection volume
m.scatter(list(df_EQ['longitude']), list(df_EQ['latitude']), latlon=True, 
          s = list((df_EQ['mag'] - 2.4)*200), alpha = 0.5, cmap='Reds')
#m.scatter(list(df['Lon']), list(df['Lat']), latlon=True, c='k')
# Create colorbar and legend
plt.title('Oklahoma Earthquakes of Magnitude 2.5 or Greater in 2017', fontsize = 14)
#
# Make legend with dummy points
for a in [20,120,220, 320, 420]:
    plt.scatter([], [], c='k', alpha=0.5, s=a,
                label=str((a/200)+2.4) + ' $Magnitude$')
plt.legend(scatterpoints=1, frameon=False,
           labelspacing=1, loc='lower left')
plt.show()
#
# PLOT - Well location, total injection volume (2011-2017), and # of >= M3.0 earthquakes
# within 20 km
sns.set(style="whitegrid")
#
# Draw the map background
fig = plt.figure(figsize=(12, 5))
m = Basemap(projection='lcc', resolution='h', 
            lat_0=35.5, lon_0=-98.6,
            width=8.5E5, height=5.0E5)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')
#
# scatter injection data, with color reflecting well type
# and size reflecting 2017 injection volume
m.scatter(list(df['Lon']), list(df['Lat']), latlon=True, 
          s = list(df['Vol_Tot']/250000), alpha = 0.5, 
          c = list(df['n_eq3_20']), cmap='Reds')
#
# Create colorbar and legend
#plt.colorbar(label=r'$Number of M3.0 and Greater Earthquakes within 20 km in 2017$')
plt.colorbar(label = 'No. of Quakes >= M3.0 within 20 km of well')
plt.clim(0, 40)
plt.title('Oklahoma Injection Well Volume (2011-2017) and Number of Earthquakes >= M3.0 within 20 km', fontsize = 14)
#
# Make legend with dummy points
for a in [1000000/250000, 20000000/250000, 40000000/250000, 60000000/250000, 80000000/250000, 100000000/250000]:
    plt.scatter([], [], c='k', alpha=0.5, s=a,
                label=str(np.round(a*250000/1000000)) + ' million barrels')
plt.legend(scatterpoints=1, frameon=False,
           labelspacing=1, loc='lower left')
plt.show()
#
# PLOT - Well location, total injection volume (2011-2017), and # of >= M4.0 earthquakes
# within 50 km
sns.set(style="whitegrid")
#
# Draw the map background
fig = plt.figure(figsize=(12, 5))
m = Basemap(projection='lcc', resolution='h', 
            lat_0=35.5, lon_0=-98.6,
            width=8.5E5, height=5.0E5)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

# scatter injection data, with color reflecting well type
# and size reflecting 2017 injection volume
m.scatter(list(df['Lon']), list(df['Lat']), latlon=True, 
          s = list(df['Vol_Tot']/250000), alpha = 0.5, 
          c = list(df['n_eq4_50']), cmap='Reds')
#
# Create colorbar and legend
#plt.colorbar(label=r'$Number of M4.0 and Greater Earthquakes within 50 km in 2017$')
plt.colorbar(label = 'No. of Quakes >= M4.0 within 50 km of well')
plt.clim(0, 4)
plt.title('Oklahoma Injection Well Volume (2011-2017) and Number of Earthquakes >= M4.0 within 50 km', fontsize = 14)
#
# Make legend with dummy points
for a in [1000000/250000, 20000000/250000, 40000000/250000, 60000000/250000, 80000000/250000, 100000000/250000]:
    plt.scatter([], [], c='k', alpha=0.5, s=a,
                label=str(np.round(a*250000/1000000)) + ' million barrels')
plt.legend(scatterpoints=1, frameon=False,
           labelspacing=1, loc='lower left')
plt.show()
#
"""# Create pairplot of well pressures from 2011 to 2017 and the number of EQ > 3.0 within 20 km
g = sns.FacetGrid(df, hue = 'WellTy', size = 4, aspect = 2)
g = (g.map(plt.scatter, "vol_wells10", "n_eq3_20", edgecolor="w").add_legend())
plt.title('Number of >M3 Earthquakes within 20 km by Injection Volume at 10 km Radius (2011-2017)')
plt.xlabel('Total Injection Vol. Within 10 km of Well, 2011-2017 [bbl]')
plt.ylabel('No. of >=M3.0 Earthquakes Within 20 km')
plt.xlim([0, 1.4e9])
plt.ylim([0,50])
plt.show()"""
#
# APPLY MACHINE LEARNING ALGORITHMS TO PREDICT NO. OF SEISMIC EVENTS
# Create training and test sets
from sklearn.model_selection import train_test_split
# Creat two different models - one to predict No. of M3.0 or greater 
# earthquakes within 20 km and # of M4.0 or greater earthquakes within 50 km
y3 = df['n_eq3_20']
y4 = df['n_eq4_50']
# Drop columns not used in the analysis
X = df.drop(['API', 'WellTy','FormName','n_eq3_20', 'n_eq4_50', 'Vol_2017',
             'Vol_2016', 'Vol_2015', 'Vol_2014', 'Vol_2013', 'Vol_2012', 
             'Vol_2011', 'vol_wells5_2017', 'vol_wells5_2016', 
             'vol_wells5_2015', 'vol_wells5_2014', 'vol_wells5_2013', 
             'vol_wells5_2012', 'vol_wells5_2011', 'vol_wells10_2017', 
             'vol_wells10_2016', 'vol_wells10_2015', 'vol_wells10_2014', 
             'vol_wells10_2013', 'vol_wells10_2012', 'vol_wells10_2011',
             'Pres_2016', 'Pres_2015', 'Pres_2014', 'Pres_2013', 'Pres_2012', 
             'Pres_2011'], axis = 1)
# Create training and test sets for well locations on responses of M3 or 
# greater earthquake within 20 km or M4 or greater earthquake within 50 km
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y3, test_size = 0.30)
X_train4, X_test4, y_train4, y_test4 = train_test_split(X, y4, test_size = 0.30)
#
# LINEAR REGRESSION
# Create the Linear Regression Training Model
#
from sklearn.linear_model import LinearRegression
lm3 = LinearRegression()
lm3.fit(X_train3,y_train3)
lm4 = LinearRegression()
lm4.fit(X_train4, y_train4)
#
def truncate(f, n):
    '''Function truncates float'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])
#
# Create predictions and plot outcome - all predictors included
pred3 = lm3.predict(X_test3)
pred4 = lm4.predict(X_test4)
#
from decimal import Decimal
MSE3 = mean_squared_error(y_test3, pred3)
MSE4 = mean_squared_error(y_test4, pred4)
R2_3 = r2_score(y_test3, pred3)
R2_4 = r2_score(y_test4, pred4)
print('\n\nLINEAR REGRESSION RESULTS')
print('------------------------------------------------------------------')
print('\nPredictors with large p values were removed from the analysis using backward elimination.')
print('\nFor linear regression to predict the no. of M3+ events within 20 km of each well using all estimators, MSE is ', ('%.2E' % Decimal(MSE3)), ' while the R2 score is ', ('%.2E' % Decimal(R2_3)),'.\n')
print('\nFor linear regression to predict the no. of M4+ events within 50 km of each well using all estimators, MSE is ', ('%.2E' % Decimal(MSE4)), ' while the R2 score is ', ('%.2E' % Decimal(R2_4)),'.\n')
#
# Plot Actual No. of Earthquakes v Predicted No. of Earthquakes - Linear Regression
print('\nLINEAR REGRESSION RESULTS PLOT (All Predictors Included):')
f = plt.figure(figsize = (12,6))
ax1 = f.add_subplot(1,2,1)
ax2 = f.add_subplot(1,2,2)
ax1.scatter(y_test3, pred3)
ax1.plot(range(0,51), ls = '--', c = 'k')
ax1.set_xlabel('Actual No. of Earthquakes M >= 3.0 within 20 km')
ax1.set_ylabel('Predicted No. of Earthquakes M >= 3.0 within 20 km')
ax1.set_xlim([0, 50])
ax1.set_ylim([0, 50])
ax2.scatter(y_test4, pred4)
ax2.plot(range(0,6), ls = '--', c = 'k')
ax2.set_xlabel('Actual No. of Earthquakes M >= 4.0 within 50 km')
ax2.set_ylabel('Predicted No. of Earthquakes M >= 4.0 within 50 km')
ax2.set_xlim([0, 5])
ax2.set_ylim([0, 5])
plt.tight_layout()
plt.show()
#
# Check Linear Regression Statistics
import statsmodels.formula.api as sm
X_opt = X.copy(deep=True)
regressor_OLS = sm.OLS(endog = y3, exog = X_opt).fit()
print('Statistical Summary: \n', regressor_OLS.summary(), '\n\n')
#
# Drop Predictors with P > 0.05 via Backward Elimination
X_opt.drop(['Deese', 'Cisco', 'Tatums', 'Simp', 'Chase', '2DCm', '2DNC',
            'Wil', 'Pont', 'Vol_Tot', '2R', 'Skinner', '2RSI', 'Penn', 
            'Calvin', 'Hoxbar', 'IBDepth', '2RIn', 'n_wells5', 'ITDepth',
            'Dutcher', 'Viola', 'Healdton', 'vol_wells5', 'n_wells10',
            'Bromide', 'Hunton'], axis = 1, inplace = True)
#
# Check for multicollinearity via VIF
#from statsmodels.stats.outliers_influence import variance_inflation_factor
#from statsmodels.tools.tools import add_constant
#
#X_collin = add_constant(X_sc)
#VIF = pd.Series([variance_inflation_factor(X_collin.values, i) for i in range(X_collin.shape[1])], index=X_collin.columns)
#print(VIF)
#    
regressor_OLS = sm.OLS(endog = y3, exog = X_opt).fit()
print('Statistical Summary (all predictors): \n', regressor_OLS.summary(), '\n\n')
#
# Re-run analysis with only significant predictors included
X_train_opt3, X_test_opt3, y_train_opt3, y_test_opt3 = train_test_split(X_opt, y3, test_size = 0.30)
X_train_opt4, X_test_opt4, y_train_opt4, y_test_opt4 = train_test_split(X_opt, y4, test_size = 0.30)
#
# Create the Linear Regression Training Model Using Only Significant Predictors
from sklearn.linear_model import LinearRegression
lm3opt = LinearRegression(normalize = True)
lm3opt.fit(X_train_opt3,y_train_opt3)
pred_opt3 = lm3opt.predict(X_test_opt3)
lm4opt = LinearRegression(normalize = True)
lm4opt.fit(X_train_opt4,y_train_opt4)
pred_opt4 = lm4opt.predict(X_test_opt4)
#
# Calculate R2 score and MSE for linear regression model with limited predictors
MSE3opt = mean_squared_error(y_test_opt3, pred_opt3)
MSE4opt = mean_squared_error(y_test_opt4, pred_opt4)
R2_3opt = r2_score(y_test_opt3, pred_opt3)
R2_4opt = r2_score(y_test_opt4, pred_opt4)
print('\nPredictors with large p values (>0.05) were removed from the analysis using backward elimination.')
print('\nFor linear regression to predict the no. of M3+ events within 20 km of each well using all estimators, MSE is ', ('%.2E' % Decimal(MSE3opt)), ' while the R2 score is ', ('%.2E' % Decimal(R2_3opt)),'.\n')
print('\nFor linear regression to predict the no. of M4+ events within 50 km of each well using all estimators, MSE is ', ('%.2E' % Decimal(MSE4opt)), ' while the R2 score is ', ('%.2E' % Decimal(R2_4opt)),'.\n')
#
# Plot Actual No. of Earthquakes v Predicted No. of Earthquakes With Some Predictors Removed - Linear Regression
print('\nLINEAR REGRESSION RESULTS PLOT (Statistically Insignicant Predictors Removed)')
f = plt.figure(figsize = (12,6))
ax1 = f.add_subplot(1,2,1)
ax2 = f.add_subplot(1,2,2)
ax1.scatter(y_test_opt3, pred_opt3)
ax1.plot(range(0,51), ls = '--', c = 'k')
ax1.set_xlabel('Actual No. of Earthquakes M >= 3.0 within 20 km')
ax1.set_ylabel('Predicted No. of Earthquakes M >= 3.0 within 20 km')
ax1.set_xlim([0, 50])
ax1.set_ylim([0, 50])
ax2.scatter(y_test_opt4, pred_opt4)
ax2.plot(range(0,6), ls = '--', c = 'k')
ax2.set_xlabel('Actual No. of Earthquakes M >= 4.0 within 50 km')
ax2.set_ylabel('Predicted No. of Earthquakes M >= 4.0 within 50 km')
ax2.set_xlim([0, 5])
ax2.set_ylim([0, 5])
plt.tight_layout()
plt.show()
#
# K NEAREST NEIGHBORS REGRESSION
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X3 = StandardScaler()
X_train_sc3 = sc_X3.fit_transform(X_train3)
X_test_sc3 = sc_X3.transform(X_test3)
sc_X4 = StandardScaler()
X_train_sc4 = sc_X4.fit_transform(X_train4)
X_test_sc4 = sc_X4.transform(X_test4)
#
# Fitting KNN to the dataset
from sklearn.neighbors import KNeighborsRegressor
"""def knn_apply(k):"""
k = 8
knn3 = KNeighborsRegressor(n_neighbors = k)
knn4 = KNeighborsRegressor(n_neighbors = k)
knn3.fit(X_train_sc3, y_train3)
knn4.fit(X_train_sc4, y_train4)
#
# Create predictions and plot outcome of SVR model
pred_knn3 = knn3.predict(X_test_sc3)
pred_knn4 = knn4.predict(X_test_sc4)
#    
# Calculate error measurements
MSE3knn = mean_squared_error(y_test3, pred_knn3)
MSE4knn = mean_squared_error(y_test4, pred_knn4)
R2_3knn = r2_score(y_test3, pred_knn3)
R2_4knn = r2_score(y_test4, pred_knn4)
"""return(MSE3knn, MSE4knn, R2_3knn, R2_4knn)"""
"""# Optimization routine to determine best K value. K = 8 found to be best
MSE3knn = []
MSE4knn = []
R2_3knn = []
R2_4knn = []
for k in range(20):
    MSE3knn.append(knn_apply(k)[0])
    MSE4knn.append(knn_apply(k)[1])
    R2_3knn.append(knn_apply(k)[2])
    R2_4knn.append(knn_apply(k)[3])
    
plt.plot(range(1,21), MSE3knn)
plt.show()

plt.plot(range(1,21), MSE4knn)
plt.show()
"""
print('\n\nK NEAREST NEIGHBOR REGRESSION RESULTS')
print('------------------------------------------------------------------')
# Calculate error measurements
MSE3knn = mean_squared_error(y_test3, pred_knn3)
MSE4knn = mean_squared_error(y_test4, pred_knn4)
R2_3knn = r2_score(y_test3, pred_knn3)
R2_4knn = r2_score(y_test4, pred_knn4)
print('\nFor KNN to predict the no. of M3+ events within 20 km of each well using all estimators, MSE is ', ('%.2E' % Decimal(MSE3knn)), ' while the R2 score is ', ('%.2E' % Decimal(R2_3knn)),'.\n')
print('\nFor KNN to predict the no. of M4+ events within 50 km of each well using all estimators, MSE is ', ('%.2E' % Decimal(MSE4knn)), ' while the R2 score is ', ('%.2E' % Decimal(R2_4knn)),'.\n')
print('\nK NEAREST NEIGHBOR REGRESSION RESULTS PLOT:')
f = plt.figure(figsize = (12,6))
ax1 = f.add_subplot(1,2,1)
ax2 = f.add_subplot(1,2,2)
ax1.scatter(y_test3, pred_knn3)
ax1.plot(range(0,51), ls = '--', c = 'k')
ax1.set_xlabel('Actual No. of Earthquakes M >= 3.0 within 20 km')
ax1.set_ylabel('Predicted No. of Earthquakes M >= 3.0 within 20 km')
ax1.set_xlim([0, 50])
ax1.set_ylim([0, 50])
ax2.scatter(y_test4, pred_knn4)
ax2.plot(range(0,6), ls = '--', c = 'k')
ax2.set_xlabel('Actual No. of Earthquakes M >= 4.0 within 50 km')
ax2.set_ylabel('Predicted No. of Earthquakes M >= 4.0 within 50 km')
ax2.set_xlim([0, 5])
ax2.set_ylim([0, 5])
plt.tight_layout()
plt.show()
#
# RANDOM FOREST REGRESSION
# Fitting RFR to the dataset
from sklearn.ensemble import RandomForestRegressor
rfr3 = RandomForestRegressor(n_estimators = 1000)
rfr3.fit(X_train_opt3, y_train_opt3)
rfr4 = RandomForestRegressor(n_estimators = 1000)
rfr4.fit(X_train_opt4, y_train_opt4)

# Create predictions and plot outcome of Random Forest model
pred_rfr3 = rfr3.predict(X_test_opt3)
pred_rfr4 = rfr4.predict(X_test_opt4)

print('\n\nRANDOM FOREST REGRESSION RESULTS')
print('------------------------------------------------------------------')

MSE3rfr = mean_squared_error(y_test_opt3, pred_rfr3)
MSE4rfr = mean_squared_error(y_test_opt4, pred_rfr4)
R2_3rfr = r2_score(y_test_opt3, pred_rfr3)
R2_4rfr = r2_score(y_test_opt4, pred_rfr4)
print('\nFor RFR to predict the no. of M3+ events within 20 km of each well using all estimators, MSE is ', ('%.2E' % Decimal(MSE3rfr)), ' while the R2 score is ', ('%.2E' % Decimal(R2_3rfr)),'.\n')
print('\nFor RFR to predict the no. of M4+ events within 50 km of each well using all estimators, MSE is ', ('%.2E' % Decimal(MSE4rfr)), ' while the R2 score is ', ('%.2E' % Decimal(R2_4rfr)),'.\n')
print('\nRANDOM FOREST REGRESSION RESULTS PLOT:')
f = plt.figure(figsize = (12,6))
ax1 = f.add_subplot(1,2,1)
ax2 = f.add_subplot(1,2,2)
ax1.scatter(y_test_opt3, pred_rfr3)
ax1.plot(range(0,51), ls = '--', c = 'k')
ax1.set_xlabel('Actual No. of Earthquakes M >= 3.0 within 20 km')
ax1.set_ylabel('Predicted No. of Earthquakes M >= 3.0 within 20 km')
ax1.set_xlim([0, 50])
ax1.set_ylim([0, 50])
ax2.scatter(y_test_opt4, pred_rfr4)
ax2.plot(range(0,6), ls = '--', c = 'k')
ax2.set_xlabel('Actual No. of Earthquakes M >= 4.0 within 50 km')
ax2.set_ylabel('Predicted No. of Earthquakes M >= 4.0 within 50 km')
ax2.set_xlim([0, 5])
ax2.set_ylim([0, 5])
plt.tight_layout()
plt.show()
#
"""#
## Cross Validation - Random Forest Regression Model Fit
scores_rfr3 = cross_val_score(estimator = rfr3, X = X_train3, y = y_train3, cv = 5)
scores_rfr4 = cross_val_score(estimator = rfr4, X = X_train4, y = y_train4, cv = 5)
print('K-Fold Cross Validation Accuracy Scores for RFR on M>=3.0 within 20 km (5 folds): %s' % scores_rfr3)
print('\n')
print('K-Fold Cross Validation Accuracy for RFR on M>=3.0 within 20 km: %.3f +/- %.3f' % (np.mean(scores_rfr3), np.std(scores_rfr3)))
print('\n')
print('K-Fold Cross Validation Accuracy Scores for RFR on M>=4.0 within 50 km (5 folds): %s' % scores_rfr4)
print('\n')
print('K-Fold Cross Validation Accuracy for RFR on M>=4.0 within 50 km: %.3f +/- %.3f' % (np.mean(scores_rfr4), np.std(scores_rfr4)))
print('\n')"""
#
# Query specific predictors and plot results
#
# Show location of  well data point 8462
well_n = 8462
sns.set(style="whitegrid")
#
# Draw the map background
fig = plt.figure(figsize=(12, 5))
m = Basemap(projection='lcc', resolution='h', 
            lat_0=35.5, lon_0=-98.6,
            width=8.5E5, height=5.0E5)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')
#
# Scatter plot of injection data, with color reflecting well type and size 
# reflecting 2017 injection volume
m.scatter(df.iloc[well_n]['Lon'], df.iloc[well_n]['Lat'], latlon=True)
""", 
          s = list(df['Vol_Tot']/250000), alpha = 0.5, 
          c = list(df['Pres_2017']), cmap='Reds')"""
#
plt.title('Location of Well Data Point 8462', fontsize = 14)
#
plt.show()
#
# Plot the effect of total injection well volume (2011-2017)
print('\nIMPACT OF VARYING PREDICTORS ON RFR RESULTS:\n')
f = plt.figure(figsize = (16,8))
ax1 = f.add_subplot(2,3,1)
ax2 = f.add_subplot(2,3,2)
ax3 = f.add_subplot(2,3,3)
ax4 = f.add_subplot(2,3,4)
ax5 = f.add_subplot(2,3,5)
ax6 = f.add_subplot(2,3,6)
#
# Plot the effect of well injection pressure (2017)
Point1 = X_opt.copy(deep=True)
pred_rfr_it = []
for pres in range(0,5500,500):
    Point1.at[well_n, 'Pres_2017'] = pres
    pred_rfr_it.append(rfr3.predict(Point1.iloc[well_n:well_n+1]))
ax1.plot(range(0,5500, 500), pred_rfr_it)
ax1.set_xlabel('Well Injection Pressure in 2017 [psi]')
ax1.set_ylabel('Pred. No. of Quakes of >= M3.0 within 20 km in 2017')
ax1.set_xlim([0,5000])
#
# Plot the effect of injection volume of all wells within 10 km of subject well (2011-2017)
Point2 = X_opt.copy(deep = True)
pred_rfr_it = []
for vol in range(0,110000000,10000000):
    Point2.at[well_n, 'vol_wells10'] = vol
    pred_rfr_it.append(rfr3.predict(Point2.iloc[well_n:well_n+1]))
ax2.plot(range(0,110000000,10000000), pred_rfr_it)
ax2.set_xlabel('Total Well Injection Volume Within 10km, 2011-2017 [bbl]')
ax2.set_ylabel('Pred. No. of Quakes of >= M3.0 within 20 km in 2017')
ax2.set_xlim([0,100000000])
#
# Plot the effect of injection depth
Point3 = X_opt.copy(deep = True)
pred_rfr_it = []
for dep in range(0,9000,500):
    Point3.at[well_n, 'TotDep'] = dep
    pred_rfr_it.append(rfr3.predict(Point3.iloc[well_n:well_n+1]))
ax3.plot(range(0,9000,500), pred_rfr_it)
ax3.set_xlabel('Well Depth [ft]')
ax3.set_ylabel('Pred. No. of Quakes of >= M3.0 within 20 km in 2017')
ax3.set_xlim([0,9500])
#
# Plot the effect of being in the Arbuckle formation
Point4 = X_opt.copy(deep = True)
pred_rfr_it = []
for a in range(0,2,1):
    Point4.at[well_n, 'Arb'] = a
    pred_rfr_it.append(rfr3.predict(Point4.iloc[well_n:well_n+1]))
ax4.plot(range(0,2,1), pred_rfr_it)
ax4.set_xlabel('Arbuckle Geologic Formation')
ax4.set_ylabel('Pred. No. of Quakes of >= M3.0 within 20 km in 2017')
ax4.set_xlim([0,1])
#
# Plot the effect of well latitude
Point5 = X_opt.copy(deep=True)
pred_rfr_it = []
for lat in np.arange(33.0, 37.1, 0.1):
    Point5.at[well_n, 'Lat'] = lat
    pred_rfr_it.append(rfr3.predict(Point5.iloc[well_n:well_n+1]))
ax5.plot(np.arange(33.,37.1,0.1), pred_rfr_it)
ax5.set_xlabel('Latitude [deg]')
ax5.set_ylabel('Pred. No. of Quakes of >= M3.0 within 20 km in 2017')
ax5.set_xlim([33,37])
#
# Plot the effect of well longitude
Point6 = X_opt.copy(deep=True)
pred_rfr_it = []
for lon in np.arange(-102.5, -94.4, 0.1):
    Point6.at[well_n, 'Lon'] = lon
    pred_rfr_it.append(rfr3.predict(Point6.iloc[well_n:well_n+1]))
ax6.plot(np.arange(-102.5, -94.4, 0.1), pred_rfr_it)
ax6.set_xlabel('Longitude [deg]')
ax6.set_ylabel('Pred. No. of Quakes of >= M3.0 within 20 km in 2017')
ax6.set_xlim([-102.5,-94.5])
plt.tight_layout()
plt.show()  