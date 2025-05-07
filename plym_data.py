import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import matplotlib.pyplot as plt
os.chdir('C:/Users/obriene/Projects/Mortality')

#Patient information by NHS or Hospital number
engine = create_engine('mssql+pyodbc://@SDMartDataLive2/PiMSMarts?'\
                       'trusted_connection=yes&driver=ODBC+Driver+17'\
                       '+for+SQL+Server')

GP_query = """SELECT [OrganisationName]
 ,[Postcode]
FROM [PiMSMarts].[OrgDataService].[PracticeCurrent]
"""
GP = pd.read_sql(GP_query, engine)
Mort_GPS = pd.read_excel('C:/Users/obriene/Projects/Mortality/Plymouth Data/GP contact spreadsheet.xlsx', usecols=[0, 1])

Mort_GPS.columns = [i.strip() for i in Mort_GPS.columns]
Mort_GPS['GP surgery'] = Mort_GPS['GP surgery'].str.strip().str.upper()

Mort_GPS = Mort_GPS.merge(GP, left_on='GP surgery', right_on='OrganisationName', how='left')
#Mort_GPS.to_csv('Plymouth Data/GP pcodes.csv', index=False)

################################################################################
#PAUSE HERE TO FILL IN MISSING POSTCODE GAPS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
################################################################################
Mort_GPS = pd.read_csv('Plymouth Data/GP pcodes.csv')[['GP surgery', 'Postcode']]
pcode_LL = pd.read_csv("G:/PerfInfo/Performance Management/PIT Adhocs/2021-2022/Hannah/Maps/pcode_LSOA_latlong.csv",
                            usecols = ['pcds', 'lsoa11']).rename(columns={'pcds':'PostCode'})
Mort_GPS['practice_postcode'] = Mort_GPS['practice_postcode'].replace('  ', ' ')
LSOAs = Mort_GPS.merge(pcode_LL, left_on='Postcode', right_on='PostCode', how='left')

#Population data
pop = pd.read_excel('C:/Users/obriene/Projects/Mortality/Plymouth Data/sapelsoabroadagetablefinal.xlsx',
                    sheet_name='Mid-2022 LSOA 2021', header=3)
pop = pop.merge(LSOAs[['GP surgery', 'lsoa11']].drop_duplicates(subset='lsoa11'), left_on='LSOA 2021 Code', right_on='lsoa11')
pop.to_excel('Plymouth Data/Community_Population_LSOA.xlsx', index=False)