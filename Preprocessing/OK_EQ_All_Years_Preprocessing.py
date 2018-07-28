"""The purpose of this pre-processing script is to prepare a dataset for 
machine learning investigation to determine the link (if any) between
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
import numpy as np
import pandas as pd
#
# ACCESS AND CLEAN 2017 WELL DATA
# Open the training data file - 2017
df_2017 = pd.read_excel('UIC injection volumes 2017.xlsx')
df_2017['Vol_2017'] = df_2017[['Jan Vol','Feb Vol','Mar Vol','Apr Vol','May Vol','Jun Vol','Jul Vol','Aug Vol','Sep Vol','Oct Vol','Nov Vol','Dec Vol']].sum(axis = 1)
df_2017['Pres_2017'] = df_2017[['Jan PSI','Feb PSI','Mar PSI','Apr PSI','May PSI','Jun PSI','Jul PSI','Aug PSI','Sep PSI','Oct PSI','Nov PSI','Dec PSI']].mean(axis = 1)
df_2017 = df_2017[(df_2017[['Vol_2017']] != 0).all(axis = 1)]
df_2017a = df_2017[['API', 'WellType', 'Lat_Y', 'Long_X', 'CountyName', 'TotalDepth', 'FormationName', 'FormationCode', 'InjTopDepth', 'InjBotDepth', 'Vol_2017', 'Pres_2017']]
#
# Drop rows without latidude/longitude data or with erroneous data (not within Oklahoma)
df_2017a = df_2017a[(df_2017a[['Lat_Y']] <= 37.0).all(axis = 1)]
df_2017a = df_2017a[(df_2017a[['Lat_Y']] >= 33.5).all(axis = 1)]
df_2017a = df_2017a[(df_2017a[['Long_X']] <= -94.0).all(axis = 1)]
df_2017a = df_2017a[(df_2017a[['Long_X']] >= -103.0).all(axis = 1)]
df_2017a = df_2017a[(df_2017a[['Pres_2017']] <= 10000.0).all(axis = 1)]
df_2017a.drop_duplicates(subset = 'API', inplace = True)
#
# ACCESS AND CLEAN 2016 WELL DATA
# Open the training data file - 2016
df_2016 = pd.read_excel('2016 1012A UIC volumes.xlsx')
df_2016['Vol_2016'] = df_2016[['Jan Vol','Feb Vol','Mar Vol','Apr Vol','May Vol','Jun Vol','Jul Vol','Aug Vol','Sep Vol','Oct Vol','Nov Vol','Dec Vol']].sum(axis = 1)
df_2016['Pres_2016'] = df_2016[['Jan PSI','Feb PSI','Mar PSI','Apr PSI','May PSI','Jun PSI','Jul PSI','Aug PSI','Sep PSI','Oct PSI','Nov PSI','Dec PSI']].mean(axis = 1)
df_2016 = df_2016[(df_2016[['Vol_2016']] != 0).all(axis = 1)]
df_2016a = df_2016[['API', 'WellType', 'Lat_Y', 'Long_X', 'CountyName', 'TotalDepth', 'FormationName', 'FormationCode', 'InjTopDepth', 'InjBotDepth', 'Vol_2016', 'Pres_2016']]
df_2016a.drop_duplicates(subset = 'API', inplace = True)
#
# Drop rows without latidude/longitude data or with erroneous data (not within Oklahoma)
df_2016a = df_2016a[(df_2016a[['Lat_Y']] <= 37.0).all(axis = 1)]
df_2016a = df_2016a[(df_2016a[['Lat_Y']] >= 33.5).all(axis = 1)]
df_2016a = df_2016a[(df_2016a[['Long_X']] <= -94.0).all(axis = 1)]
df_2016a = df_2016a[(df_2016a[['Long_X']] >= -103.0).all(axis = 1)]
df_2016a = df_2016a[(df_2016a[['Pres_2016']] <= 10000.0).all(axis = 1)]
df_2016a.drop_duplicates(subset = 'API', inplace = True)
#
# ACCESS AND CLEAN 2015 WELL DATA
# Open the training data file - 2015
df_2015 = pd.read_excel('2015 1012A UIC volumes.xlsx')
df_2015['Vol_2015'] = df_2015[['Jan Vol','Feb Vol','Mar Vol','Apr Vol','May Vol','Jun Vol','Jul Vol','Aug Vol','Sep Vol','Oct Vol','Nov Vol','Dec Vol']].sum(axis = 1)
df_2015['Pres_2015'] = df_2015[['Jan PSI','Feb PSI','Mar PSI','Apr PSI','May PSI','Jun PSI','Jul PSI','Aug PSI','Sep PSI','Oct PSI','Nov PSI','Dec PSI']].mean(axis = 1)
df_2015 = df_2015[(df_2015[['Vol_2015']] != 0).all(axis = 1)]
df_2015a = df_2015[['API', 'WellType', 'Lat_Y', 'Long_X', 'CountyName', 'TotalDepth', 'FormationName', 'FormationCode', 'InjTopDepth', 'InjBotDepth', 'Vol_2015', 'Pres_2015']]
df_2015a.drop_duplicates(subset = 'API', inplace = True)
#
# Drop rows without latidude/longitude data or with erroneous data (not within Oklahoma)
df_2015a = df_2015a[(df_2015a[['Lat_Y']] <= 37.0).all(axis = 1)]
df_2015a = df_2015a[(df_2015a[['Lat_Y']] >= 33.5).all(axis = 1)]
df_2015a = df_2015a[(df_2015a[['Long_X']] <= -94.0).all(axis = 1)]
df_2015a = df_2015a[(df_2015a[['Long_X']] >= -103.0).all(axis = 1)]
df_2015a = df_2015a[(df_2015a[['Pres_2015']] <= 10000.0).all(axis = 1)]
df_2015a.drop_duplicates(subset = 'API', inplace = True)
#
# ACCESS AND CLEAN 2014 WELL DATA
# Open the training data file - 2014
df_2014 = pd.read_excel('2014 1012A UIC volumes.xlsx')
df_2014['Vol_2014'] = df_2014[['Jan Vol','Feb Vol','Mar Vol','Apr Vol','May Vol','Jun Vol','Jul Vol','Aug Vol','Sep Vol','Oct Vol','Nov Vol','Dec Vol']].sum(axis = 1)
df_2014['Pres_2014'] = df_2014[['Jan PSI','Feb PSI','Mar PSI','Apr PSI','May PSI','Jun PSI','Jul PSI','Aug PSI','Sep PSI','Oct PSI','Nov PSI','Dec PSI']].mean(axis = 1)
df_2014 = df_2014[(df_2014[['Vol_2014']] != 0).all(axis = 1)]
df_2014a = df_2014[['API', 'WellType', 'Lat_Y', 'Long_X', 'CountyName', 'TotalDepth', 'FormationName', 'FormationCode', 'InjTopDepth', 'InjBotDepth', 'Vol_2014', 'Pres_2014']]
df_2014a.drop_duplicates(subset = 'API', inplace = True)
#
# Drop rows without latidude/longitude data or with erroneous data (not within Oklahoma)
df_2014a = df_2014a[(df_2014a[['Lat_Y']] <= 37.0).all(axis = 1)]
df_2014a = df_2014a[(df_2014a[['Lat_Y']] >= 33.5).all(axis = 1)]
df_2014a = df_2014a[(df_2014a[['Long_X']] <= -94.0).all(axis = 1)]
df_2014a = df_2014a[(df_2014a[['Long_X']] >= -103.0).all(axis = 1)]
df_2014a = df_2014a[(df_2014a[['Pres_2014']] <= 10000.0).all(axis = 1)]
df_2014a.drop_duplicates(subset = 'API', inplace = True)
#
# ACCESS AND CLEAN 2013 WELL DATA
# Open the training data file - 2013
df_2013 = pd.read_excel('2013 1012A UIC volumes.xlsx')
df_2013['Vol_2013'] = df_2013[['Jan Vol','Feb Vol','Mar Vol','Apr Vol','May Vol','Jun Vol','Jul Vol','Aug Vol','Sep Vol','Oct Vol','Nov Vol','Dec Vol']].sum(axis = 1)
df_2013['Pres_2013'] = df_2013[['Jan PSI','Feb PSI','Mar PSI','Apr PSI','May PSI','Jun PSI','Jul PSI','Aug PSI','Sep PSI','Oct PSI','Nov PSI','Dec PSI']].mean(axis = 1)
df_2013 = df_2013[(df_2013[['Vol_2013']] != 0).all(axis = 1)]
df_2013a = df_2013[['API', 'WellType', 'Lat_Y', 'Long_X', 'CountyName', 'TotalDepth', 'FormationName', 'FormationCode', 'InjTopDepth', 'InjBotDepth', 'Vol_2013', 'Pres_2013']]
df_2013a.drop_duplicates(subset = 'API', inplace = True)
#
# Drop rows without latidude/longitude data or with erroneous data (not within Oklahoma)
df_2013a = df_2013a[(df_2013a[['Lat_Y']] <= 37.0).all(axis = 1)]
df_2013a = df_2013a[(df_2013a[['Lat_Y']] >= 33.5).all(axis = 1)]
df_2013a = df_2013a[(df_2013a[['Long_X']] <= -94.0).all(axis = 1)]
df_2013a = df_2013a[(df_2013a[['Long_X']] >= -103.0).all(axis = 1)]
df_2013a = df_2013a[(df_2013a[['Pres_2013']] <= 10000.0).all(axis = 1)]
df_2013a.drop_duplicates(subset = 'API', inplace = True)
#
# ACCESS AND CLEAN 2012 WELL DATA
# Open the training data file - 2012
df_2012 = pd.read_excel('2012 1012A UIC volumes.xlsx')
df_2012['Vol_2012'] = df_2012[['Jan Vol','Feb Vol','Mar Vol','Apr Vol','May Vol','Jun Vol','Jul Vol','Aug Vol','Sep Vol','Oct Vol','Nov Vol','Dec Vol']].sum(axis = 1)
df_2012['Pres_2012'] = df_2012[['Jan PSI','Feb PSI','Mar PSI','Apr PSI','May PSI','Jun PSI','Jul PSI','Aug PSI','Sep PSI','Oct PSI','Nov PSI','Dec PSI']].mean(axis = 1)
df_2012 = df_2012[(df_2012[['Vol_2012']] != 0).all(axis = 1)]
df_2012a = df_2012[['API', 'WellType', 'Lat_Y', 'Long_X', 'CountyName', 'TotalDepth', 'FormationName', 'FormationCode', 'InjTopDepth', 'InjBotDepth', 'Vol_2012', 'Pres_2012']]
df_2012a.drop_duplicates(subset = 'API', inplace = True)
#
# Drop rows without latidude/longitude data or with erroneous data (not within Oklahoma)
df_2012a = df_2012a[(df_2012a[['Lat_Y']] <= 37.0).all(axis = 1)]
df_2012a = df_2012a[(df_2012a[['Lat_Y']] >= 33.5).all(axis = 1)]
df_2012a = df_2012a[(df_2012a[['Long_X']] <= -94.0).all(axis = 1)]
df_2012a = df_2012a[(df_2012a[['Long_X']] >= -103.0).all(axis = 1)]
df_2012a = df_2012a[(df_2012a[['Pres_2012']] <= 10000.0).all(axis = 1)]
df_2012a.drop_duplicates(subset = 'API', inplace = True)
#
# ACCESS AND CLEAN 2011 WELL DATA
# Open the training data file - 2011
df_2011 = pd.read_excel('2011 1012A UIC volumes.xlsx')
df_2011['Vol_2011'] = df_2011[['Jan Vol','Feb Vol','Mar Vol','Apr Vol','May Vol','Jun Vol','Jul Vol','Aug Vol','Sep Vol','Oct Vol','Nov Vol','Dec Vol']].sum(axis = 1)
df_2011['Pres_2011'] = df_2011[['Jan PSI','Feb PSI','Mar PSI','Apr PSI','May PSI','Jun PSI','Jul PSI','Aug PSI','Sep PSI','Oct PSI','Nov PSI','Dec PSI']].mean(axis = 1)
df_2011 = df_2011[(df_2011[['Vol_2011']] != 0).all(axis = 1)]
df_2011a = df_2011[['API', 'WellType', 'Lat_Y', 'Long_X', 'CountyName', 'TotalDepth', 'FormationName', 'FormationCode', 'InjTopDepth', 'InjBotDepth', 'Vol_2011', 'Pres_2011']]
df_2011a.drop_duplicates(subset = 'API', inplace = True)
#
# Drop rows without latidude/longitude data or with erroneous data (not within Oklahoma)
df_2011a = df_2011a[(df_2011a[['Lat_Y']] <= 37.0).all(axis = 1)]
df_2011a = df_2011a[(df_2011a[['Lat_Y']] >= 33.5).all(axis = 1)]
df_2011a = df_2011a[(df_2011a[['Long_X']] <= -94.0).all(axis = 1)]
df_2011a = df_2011a[(df_2011a[['Long_X']] >= -103.0).all(axis = 1)]
df_2011a = df_2011a[(df_2011a[['Pres_2011']] <= 10000.0).all(axis = 1)]
df_2011a.drop_duplicates(subset = 'API', inplace = True)
#
# Assemble data for 2011 through 2017
df_full = df_2017a.set_index('API').join(df_2016a.set_index('API'), how = 'outer', lsuffix = '2017', rsuffix = '2016').join(df_2015a.set_index('API'), how = 'outer', rsuffix = '2015').join(df_2014a.set_index('API'), how = 'outer', rsuffix = '2014').join(df_2013a.set_index('API'), how = 'outer', rsuffix = '2013').join(df_2012a.set_index('API'), how = 'outer', rsuffix = '2012').join(df_2011a.set_index('API'), how = 'outer', rsuffix = '2011')
df_full['Lat'] = df_full['Lat_Y2017'].combine_first(df_full['Lat_Y2016']).combine_first(df_full['Lat_Y']).combine_first(df_full['Lat_Y2014']).combine_first(df_full['Lat_Y2013']).combine_first(df_full['Lat_Y2012']).combine_first(df_full['Lat_Y2011'])
df_full['Lon'] = df_full['Long_X2017'].combine_first(df_full['Long_X2016']).combine_first(df_full['Long_X']).combine_first(df_full['Long_X2014']).combine_first(df_full['Long_X2013']).combine_first(df_full['Long_X2012']).combine_first(df_full['Long_X2011'])
df_full['WellTy'] = df_full['WellType2017'].combine_first(df_full['WellType2016']).combine_first(df_full['WellType']).combine_first(df_full['WellType2014']).combine_first(df_full['WellType2013']).combine_first(df_full['WellType2012']).combine_first(df_full['WellType2011'])
df_full['TotDep'] = df_full[['TotalDepth2017', 'TotalDepth2016','TotalDepth','TotalDepth2014','TotalDepth2013','TotalDepth2012','TotalDepth2011']].max(axis=1)
df_full['FormName'] = df_full['FormationName2017'].combine_first(df_full['FormationName2016']).combine_first(df_full['FormationName']).combine_first(df_full['FormationName2014']).combine_first(df_full['FormationName2013']).combine_first(df_full['FormationName2012']).combine_first(df_full['FormationName2011'])
df_full['ITDepth'] = df_full[['InjTopDepth2017', 'InjTopDepth2016','InjTopDepth','InjTopDepth2014','InjTopDepth2013','InjTopDepth2012','InjTopDepth2011']].max(axis=1)
df_full['IBDepth'] = df_full[['InjBotDepth2017', 'InjBotDepth2016','InjBotDepth','InjBotDepth2014','InjBotDepth2013','InjBotDepth2012','InjBotDepth2011']].max(axis=1)
df_fulla = df_full[['Lat', 'Lon', 'WellTy', 'TotDep', 'FormName', 'ITDepth', 'IBDepth', 'Vol_2017', 'Vol_2016', 'Vol_2015', 'Vol_2014', 'Vol_2013', 'Vol_2012', 'Vol_2011', 'Pres_2017', 'Pres_2016', 'Pres_2015', 'Pres_2014', 'Pres_2013', 'Pres_2012', 'Pres_2011']]
#
# Fill in missing pressure and volume data with zero (missing data in these means no record - well not active)
df_fulla['Vol_2017'].fillna(0, inplace = True)
df_fulla['Vol_2016'].fillna(0, inplace = True)
df_fulla['Vol_2015'].fillna(0, inplace = True)
df_fulla['Vol_2014'].fillna(0, inplace = True)
df_fulla['Vol_2013'].fillna(0, inplace = True)
df_fulla['Vol_2012'].fillna(0, inplace = True)
df_fulla['Vol_2011'].fillna(0, inplace = True)
df_fulla['Pres_2017'].fillna(0, inplace = True)
df_fulla['Pres_2016'].fillna(0, inplace = True)
df_fulla['Pres_2015'].fillna(0, inplace = True)
df_fulla['Pres_2014'].fillna(0, inplace = True)
df_fulla['Pres_2013'].fillna(0, inplace = True)
df_fulla['Pres_2012'].fillna(0, inplace = True)
df_fulla['Pres_2011'].fillna(0, inplace = True)
#
# Fill in missing well depth data using mean of other wells
total_depth = df_fulla['TotDep'].mean()
top_depth = df_fulla['ITDepth'].mean()
bottom_depth = df_fulla['IBDepth'].mean()
def total_depth_calc(cols):
    """ This function fills in missing depth data with mean"""
    depth = cols[0]
    if pd.isnull(depth):
        return total_depth
    else:
        return depth
#
def top_depth_calc(cols):
    """ This function fills in missing depth data with mean"""
    depth = cols[0]
    if pd.isnull(depth):
        return top_depth
    else:
        return depth
#
def bottom_depth_calc(cols):
    """ This function fills in missing depth data with mean"""
    depth = cols[0]
    if pd.isnull(depth):
        return bottom_depth
    else:
        return depth
# 
df_fulla['TotDep'] = df_fulla[['TotDep']].apply(total_depth_calc, axis=1)  
df_fulla['ITDepth'] = df_fulla[['ITDepth']].apply(top_depth_calc, axis=1)   
df_fulla['IBDepth'] = df_fulla[['IBDepth']].apply(bottom_depth_calc, axis=1)
#
# Import data test categorical columns - Well Type
well_type = pd.get_dummies(df_fulla['WellTy'], drop_first = True)
df_fulla = pd.concat([df_fulla, well_type], axis=1)
#
# CREATE BINARY MARKERS FOR INJECTION FORMATION NAMES - for all formations with
# at least 100 wells, create predictor for formation name
# Insert column to check if in Arbuckle Formation
def isarb(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'arbuckle' in position.lower():
        return 1
    else:
        return 0
df_fulla['Arb'] = df_fulla[['FormName']].apply(isarb, axis = 1)
#
# Insert column to check if in Bartlesville Formation
def isbart(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'bartle' in position.lower():
        return 1
    else:
        return 0
df_fulla['Bart'] = df_fulla[['FormName']].apply(isbart, axis = 1)

# Insert column to check if in Booch Formation
def isbooch(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'booch' in position.lower():
        return 1
    else:
        return 0
df_fulla['Booch'] = df_fulla[['FormName']].apply(isbooch, axis = 1)

# Insert column to check if in Bromide Formation
def isbrom(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'bromide' in position.lower():
        return 1
    else:
        return 0
df_fulla['Bromide'] = df_fulla[['FormName']].apply(isbrom, axis = 1)

# Insert column to check if in Burgess Formation
def isburg(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'burgess' in position.lower():
        return 1
    else:
        return 0
df_fulla['Burgess'] = df_fulla[['FormName']].apply(isburg, axis = 1)

# Insert column to check if in Calvin Formation
def iscalv(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'calvin' in position.lower():
        return 1
    else:
        return 0
df_fulla['Calvin'] = df_fulla[['FormName']].apply(iscalv, axis = 1)

# Insert column to check if in Chase Formation
def ischase(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'chase' in position.lower():
        return 1
    else:
        return 0
df_fulla['Chase'] = df_fulla[['FormName']].apply(ischase, axis = 1)

# Insert column to check if in Cisco Formation
def iscisco(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'cisco' in position.lower():
        return 1
    else:
        return 0
df_fulla['Cisco'] = df_fulla[['FormName']].apply(iscisco, axis = 1)

# Insert column to check if in Cleveland Formation
def isclev(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'cleveland' in position.lower():
        return 1
    else:
        return 0
df_fulla['Clev'] = df_fulla[['FormName']].apply(isclev, axis = 1)

# Insert column to check if in Cromwell Formation
def iscrom(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'cromwell' in position.lower():
        return 1
    else:
        return 0
df_fulla['Crom'] = df_fulla[['FormName']].apply(iscrom, axis = 1)

# Insert column to check if in Deese Formation
def isdeese(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'dees' in position.lower():
        return 1
    else:
        return 0
df_fulla['Deese'] = df_fulla[['FormName']].apply(isdeese, axis = 1)

# Insert column to check if in Dutcher Formation
def isdutcher(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'dutcher' in position.lower():
        return 1
    else:
        return 0
df_fulla['Dutcher'] = df_fulla[['FormName']].apply(isdutcher, axis = 1)

# Insert column to check if in Gilcrease Formation
def isgil(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'gilcrease' in position.lower():
        return 1
    else:
        return 0
df_fulla['Gilcrease'] = df_fulla[['FormName']].apply(isgil, axis = 1)

# Insert column to check if in Healdton Formation
def isheald(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'healdton' in position.lower():
        return 1
    else:
        return 0
df_fulla['Healdton'] = df_fulla[['FormName']].apply(isheald, axis = 1)

# Insert column to check if in Hoxbar Formation
def ishox(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'hoxbar' in position.lower():
        return 1
    else:
        return 0
df_fulla['Hoxbar'] = df_fulla[['FormName']].apply(ishox, axis = 1)

# Insert column to check if in Hunton Formation
def ishunt(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'hunton' in position.lower():
        return 1
    else:
        return 0
df_fulla['Hunton'] = df_fulla[['FormName']].apply(ishunt, axis = 1)

# Insert column to check if in Layton Formation
def islayton(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'layton' in position.lower():
        return 1
    else:
        return 0
df_fulla['Layton'] = df_fulla[['FormName']].apply(islayton, axis = 1)

# Insert column to check if in Loco Formation
def isloco(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'loco' in position.lower():
        return 1
    else:
        return 0
df_fulla['Loco'] = df_fulla[['FormName']].apply(isloco, axis = 1)

# Insert column to check if in Morrow Formation
def ismorrow(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'morrow' in position.lower():
        return 1
    else:
        return 0
df_fulla['Morrow'] = df_fulla[['FormName']].apply(ismorrow, axis = 1)

# Insert column to check if in Penn Formation
def ispenn(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'penn' in position.lower():
        return 1
    else:
        return 0
df_fulla['Penn'] = df_fulla[['FormName']].apply(ispenn, axis = 1)

# Insert column to check if in Permian Formation
def ispermian(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'permian' in position.lower():
        return 1
    else:
        return 0
df_fulla['Permian'] = df_fulla[['FormName']].apply(ispermian, axis = 1)

# Insert column to check if in Pontotoc Formation
def ispont(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'pontotoc' in position.lower():
        return 1
    else:
        return 0
df_fulla['Pont'] = df_fulla[['FormName']].apply(ispont, axis = 1)

# Insert column to check if in Prue Formation
def isprue(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'prue' in position.lower():
        return 1
    else:
        return 0
df_fulla['Prue'] = df_fulla[['FormName']].apply(isprue, axis = 1)

# Insert column to check if in Red Fork Formation
def isred(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif ('red fork' in position.lower()) or ('redfork' in position.lower()):
        return 1
    else:
        return 0
df_fulla['Red'] = df_fulla[['FormName']].apply(isred, axis = 1)

# Insert column to check if in Simpson Formation
def issimp(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif ('simpson' in position.lower()):
        return 1
    else:
        return 0
df_fulla['Simp'] = df_fulla[['FormName']].apply(issimp, axis = 1)

# Insert column to check if in Skinner Formation
def isskin(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif ('skinner' in position.lower()):
        return 1
    else:
        return 0
df_fulla['Skinner'] = df_fulla[['FormName']].apply(isskin, axis = 1)

# Insert column to check if in Tatums Formation
def istat(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif ('tatums' in position.lower()):
        return 1
    else:
        return 0
df_fulla['Tatums'] = df_fulla[['FormName']].apply(istat, axis = 1)

# Insert column to check if in Viola Formation
def isviola(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif ('viola' in position.lower()):
        return 1
    else:
        return 0
df_fulla['Viola'] = df_fulla[['FormName']].apply(isviola, axis = 1)

# Insert column to check if in Wilcox Formation
def iswil(cols):
    position = cols[0]
    if pd.isnull(position):
        return 0
    elif 'wilcox' in position.lower():
        return 1
    else:
        return 0
df_fulla['Wil'] = df_fulla[['FormName']].apply(iswil, axis = 1)

#
# Count number of wells within 5 km or 10 km of each well, as well as the
# total injection volume within 5km or 10 km of each well for each recorded
# year
n_wells5 = []
n_wells10 = []
vol_tot_5_2017 = []
vol_tot_10_2017 = []
vol_tot_5_2016 = []
vol_tot_10_2016 = []
vol_tot_5_2015 = []
vol_tot_10_2015 = []
vol_tot_5_2014 = []
vol_tot_10_2014 = []
vol_tot_5_2013 = []
vol_tot_10_2013 = []
vol_tot_5_2012 = []
vol_tot_10_2012 = []
vol_tot_5_2011 = []
vol_tot_10_2011 = []
for i in range(len(df_fulla)):
    count5 = 0
    count10 = 0
    vol5_2017 =0
    vol10_2017 = 0
    vol5_2016 = 0
    vol10_2016 = 0
    vol5_2015 = 0
    vol10_2015 = 0
    vol5_2014 = 0
    vol10_2014 = 0
    vol5_2013 = 0
    vol10_2013 = 0
    vol5_2012 = 0
    vol10_2012 = 0
    vol5_2011 = 0
    vol10_2011 = 0
    for j in range(len(df_fulla)):
        lat_1 = np.radians(df_fulla['Lat'].iloc[i])
        lat_2 = np.radians(df_fulla['Lat'].iloc[j])
        lon_1 = np.radians(df_fulla['Lon'].iloc[i])
        lon_2 = np.radians(df_fulla['Lon'].iloc[j])
        if lat_1 == lat_2 and lon_1 == lon_2:
            d = 0
        else:    
            a = np.power(np.sin(lat_1 - lat_2)/2,2) + np.cos(lat_1)*np.cos(lat_2)*np.power(np.sin(lon_1 - lon_2)/2,2)
            c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
            d = 6371 * c
        if d <=5:
            count5 += 1
            vol5_2017 += df_fulla['Vol_2017'].iloc[j]
            vol5_2016 += df_fulla['Vol_2016'].iloc[j]
            vol5_2015 += df_fulla['Vol_2015'].iloc[j]
            vol5_2014 += df_fulla['Vol_2014'].iloc[j]
            vol5_2013 += df_fulla['Vol_2013'].iloc[j]
            vol5_2012 += df_fulla['Vol_2012'].iloc[j]
            vol5_2011 += df_fulla['Vol_2011'].iloc[j]
        if d <= 10:
            count10 += 1
            vol10_2017 += df_fulla['Vol_2017'].iloc[j]
            vol10_2016 += df_fulla['Vol_2016'].iloc[j]
            vol10_2015 += df_fulla['Vol_2015'].iloc[j]
            vol10_2014 += df_fulla['Vol_2014'].iloc[j]
            vol10_2013 += df_fulla['Vol_2013'].iloc[j]
            vol10_2012 += df_fulla['Vol_2012'].iloc[j]
            vol10_2011 += df_fulla['Vol_2011'].iloc[j]
    n_wells5.append(count5)
    n_wells10.append(count10)
    vol_tot_5_2017.append(vol5_2017)
    vol_tot_10_2017.append(vol10_2017)
    vol_tot_5_2016.append(vol5_2016)
    vol_tot_10_2016.append(vol10_2016)
    vol_tot_5_2015.append(vol5_2015)
    vol_tot_10_2015.append(vol10_2015)
    vol_tot_5_2014.append(vol5_2014)
    vol_tot_10_2014.append(vol10_2014)
    vol_tot_5_2013.append(vol5_2013)
    vol_tot_10_2013.append(vol10_2013)
    vol_tot_5_2012.append(vol5_2012)
    vol_tot_10_2012.append(vol10_2012)
    vol_tot_5_2011.append(vol5_2011)
    vol_tot_10_2011.append(vol10_2011)
#    
df_fulla['n_wells5'] = n_wells5
df_fulla['n_wells10'] = n_wells10
df_fulla['vol_wells5_2017'] = vol_tot_5_2017
df_fulla['vol_wells10_2017'] = vol_tot_10_2017
df_fulla['vol_wells5_2016'] = vol_tot_5_2016
df_fulla['vol_wells10_2016'] = vol_tot_10_2016
df_fulla['vol_wells5_2015'] = vol_tot_5_2015
df_fulla['vol_wells10_2015'] = vol_tot_10_2015
df_fulla['vol_wells5_2014'] = vol_tot_5_2014
df_fulla['vol_wells10_2014'] = vol_tot_10_2014
df_fulla['vol_wells5_2013'] = vol_tot_5_2013
df_fulla['vol_wells10_2013'] = vol_tot_10_2013
df_fulla['vol_wells5_2012'] = vol_tot_5_2012
df_fulla['vol_wells10_2012'] = vol_tot_10_2012
df_fulla['vol_wells5_2011'] = vol_tot_5_2011
df_fulla['vol_wells10_2011'] = vol_tot_10_2011
#
# Read in USGS earthquake record data for all earthquakes occurring in 
# Oklahoma of magnitude 2.5 or greater during 2017
df_eq = pd.read_csv('OK_EQ.csv')
# Count number of earthquakes 3M, 20 km away; and 4M, 50 km away
n_eq3_20 = []
n_eq4_50 = []
for i in range(len(df_fulla)):
    count3 = 0
    count4 = 0
    count5 = 0
    for j in range(len(df_eq)):
        lat_1 = np.radians(df_fulla['Lat'].iloc[i])
        lat_2 = np.radians(df_eq['latitude'].iloc[j])
        lon_1 = np.radians(df_fulla['Lon'].iloc[i])
        lon_2 = np.radians(df_eq['longitude'].iloc[j])
        mag = df_eq['mag'].iloc[j]
        #
        # Calculate distance between well and epicenter
        a = np.power(np.sin(lat_1 - lat_2)/2,2) + np.cos(lat_1)*np.cos(lat_2)*np.power(np.sin(lon_1 - lon_2)/2,2)
        c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d = 6371 * c
        if d <= 20 and mag >= 3:
            count3 += 1
        if d <= 50 and mag >= 4:
            count4 += 1
    n_eq3_20.append(count3)
    n_eq4_50.append(count4)
#    
df_fulla['n_eq3_20'] = n_eq3_20
df_fulla['n_eq4_50'] = n_eq4_50
#
# Write the cleaned data to file
df_fulla.to_csv('full_data.csv')