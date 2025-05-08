import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from scipy.stats import mannwhitneyu
from scipy.stats import chisquare
import itertools
import seaborn as sns
import os
import matplotlib.pyplot as plt
os.chdir('C:/Users/obriene/Projects/Inequalities/Mortality')

# =============================================================================
#     #Read in and tidy data
# =============================================================================
def data_tidy(df, lookup, merge_col):
    #remove any empty columns and strip out newlines and merge onto patient
    #lookup
    df = df.dropna(axis=1, how='all')
    df = df.drop([col for col in df.columns if 'Unnamed' in col], axis=1)
    df.columns = [col.strip().replace('\n', '') for col in df.columns]
    #Merge data on both hospital and NHS number as both seem to be in columns.
    hosp_no_merge = df.merge(lookup.dropna(subset='Hospital Number')
                             .drop_duplicates(subset='Hospital Number'),
                             left_on=merge_col, right_on='Hospital Number',
                             how='left')
    nhs_no_merge = df.merge(lookup.dropna(subset='NHS Number')
                            .drop_duplicates('NHS Number'),
                            left_on=merge_col, right_on='NHS Number',
                            how='left')
    cols = lookup.columns[2:]
    df[cols] = hosp_no_merge[cols].fillna(nhs_no_merge[cols])
    return df

##########Patient information by NHS or Hospital number
engine = create_engine('mssql+pyodbc://@SDMartDataLive2/PiMSMarts?'\
                       'trusted_connection=yes&driver=ODBC+Driver+17'\
                       '+for+SQL+Server')
pat_lookup_query = """
SET NOCOUNT ON
SELECT nhs_number
,pasid
,pat_dob
,sexxx
,ethgr
,IndexValue
,pat_address1
,pat_address2
,pat_pcode
INTO #pat
FROM [PiMSMarts].[dbo].[patients]
LEFT JOIN [Reference].[vw_IndicesOfMultipleDeprivation2019_DecileByPostcode] AS IMD
ON pat_pcode = PostcodeFormatted
WHERE pat_dob > DATEADD(YEAR, -116, getdate())

SELECT
pat.nhs_number as [NHS Number]
,pat.pasid as [Hospital Number]
,pat.pat_dob
,pat.pat_pcode
,pat.sexxx
,pat.ethgr
,pat.IndexValue as IMD
,carehome = case when val.valid_ch = 'y' then 'Y' else 'N' end
from #pat pat
left  join infodb.dbo.cset_care_homes chomes
on replace(pat.pat_pcode , ' ', '')=replace(chomes.postcode , ' ', '')
left   join infodb.dbo.nursing_home_validation val
on isnull(pat.pat_address1,'')+isnull(pat.pat_address2,'')=isnull(val.pat_address1,'')+isnull(val.pat_address2,'')
and pat.pat_pcode=val.pat_pcode
and chomes.ResidentialHome=val.residentialhome
and val.valid_ch = 'y'
"""
pat_lookup = pd.read_sql(pat_lookup_query, engine)
pat_lookup['IMD'] = pat_lookup['IMD'].astype(float)
pat_lookup['IMD_split'] = np.where(pat_lookup['IMD'] <=2, 'IMD 1-2', 'IMD 3-10')
pat_lookup['Ethnicity'] = np.where(pat_lookup['ethgr']=='A',
                                   'White British', 'Ethnic Minority')
engine.dispose()

##########Deaths data
deaths_filepath = 'G:/MEOffice/1. Bereavement Register/2022/'
#####Community
comm_deaths = data_tidy(pd.read_excel(deaths_filepath
              + 'Community Bereavement Register/Community Bereavement Register.xlsx',
              sheet_name='Community Deaths'),
              pat_lookup.dropna(subset='NHS Number'),
              'NHS Number').drop_duplicates(subset='NHS Number')
comm_deaths['Age of Death'] = ((comm_deaths['Date of Death']
                                - comm_deaths['pat_dob']).dt.days/365).round()
#Fill in missing postcodes and get stem
comm_deaths['Postcode of Deceased'] = (comm_deaths['Postcode of Deceased']
                                       .fillna(comm_deaths['pat_pcode']))
comm_deaths['pcode_stem'] = comm_deaths['Postcode of Deceased'].str[:-3].str.replace(' ', '')
comm_CH_deaths = comm_deaths.loc[comm_deaths['carehome'] == 'Y'].copy()
comm_oth_deaths = comm_deaths.loc[comm_deaths['carehome'] == 'N'].copy()
#####Hospital
hosp_deaths = data_tidy(pd.read_excel(deaths_filepath + 'Register 2022.xlsx',
              sheet_name='2022'), pat_lookup, 'Hospital Number')
hosp_deaths['ED Death'] = np.where(hosp_deaths['Ward']=='ED', 'Y', 'N')
hosp_deaths['Age of Death'] = ((hosp_deaths['Date of Death']
                                - hosp_deaths['pat_dob']).dt.days/365).round()
#Fill in missing postcodes and get stem
hosp_deaths['Postcode Deceased'] = (hosp_deaths['Postcode Deceased']
                                      .fillna(hosp_deaths['pat_pcode']))
hosp_deaths['pcode_stem'] = hosp_deaths['Postcode Deceased'].str[:-3].str.replace(' ', '')
#Filter to patients that match the community data (so comparing like for like)
hosp_deaths = hosp_deaths.loc[hosp_deaths['pcode_stem'].isin(comm_deaths['pcode_stem'].drop_duplicates())].copy()
hosp_ED_deaths = hosp_deaths.loc[hosp_deaths['ED Death'] == 'Y'].copy()
hosp_oth_deaths = hosp_deaths.loc[hosp_deaths['ED Death'] == 'N'].copy()
#####Neoantal
baby_deaths = data_tidy(pd.read_excel(deaths_filepath + 'Register 2022.xlsx',
              sheet_name='Baby Deaths')
              .rename(columns={'Mothers HN':'Hospital Number'}),
              pat_lookup, 'Hospital Number')
baby_deaths['pcode_stem'] = baby_deaths['pat_pcode'].str[:-3].str.replace(' ', '')
baby_deaths = baby_deaths.loc[baby_deaths['pcode_stem'].isin(comm_deaths['pcode_stem'].drop_duplicates())].copy()
#####list of all ages
max_ages = max(hosp_deaths['Age of Death'].max(), comm_deaths['Age of Death'].max())
all_ages = [i for i in range(int(max_ages+1))]

###########All Deaths Population Data
population = pd.concat([comm_deaths, hosp_deaths, baby_deaths])
population['count'] = 1

# =============================================================================
#     #PLOTS
# =============================================================================
###################Proportion of each population by IMD plot####################
#######Calculations
def IMD_dist(df, col):
    group = (df.groupby('IMD', as_index=False)[col].count()
             .rename(columns={col:'Count'}).astype(int).sort_values(by='IMD'))
    group['Proportion'] = group['Count']/group['Count'].sum()
    return group
#Get the distributions for each group
pop_IMD_dist = IMD_dist(population, 'count')
baby_IMD_dist = IMD_dist(baby_deaths, 'Hospital Number')
comm_CH_IMD_dist = IMD_dist(comm_CH_deaths, 'NHS Number')
comm_oth_IMD_dist = IMD_dist(comm_oth_deaths, 'NHS Number')
hosp_ED_IMD_dist = IMD_dist(hosp_ED_deaths, 'Hospital Number')
hosp_oth_IMD_dist = IMD_dist(hosp_oth_deaths, 'Hospital Number')

#list of IMD numbers for xticks
IMDs = (pat_lookup['IMD'].drop_duplicates().dropna().sort_values()
        .astype(int).values)
#######Plot
fig, axs = plt.subplots(3, 2)
#Loop through and plot each group as a subplot.
subplots = [(axs[0, 0], pop_IMD_dist, population['IMD'], 'All Deaths'),
            (axs[0, 1], baby_IMD_dist, baby_deaths['IMD'], 'Baby Deaths'),
            (axs[1, 0], comm_CH_IMD_dist, comm_CH_deaths['IMD'], 'Community Carehome Deaths'),
            (axs[1, 1], comm_oth_IMD_dist, comm_oth_deaths['IMD'], 'Community Non-Carehome Deaths'),
            (axs[2, 0], hosp_ED_IMD_dist, hosp_ED_deaths['IMD'], 'ED Hospital Deaths'),
            (axs[2, 1], hosp_oth_IMD_dist, hosp_oth_deaths['IMD'], 'Other Hospital Deaths')]
for axis, dist, all, title in subplots:
    #bar chart
    axis.bar(dist['IMD'].values, dist['Proportion'].values, color='cornflowerblue')
    #LQ, median and UQ lines
    median = all.median()
    LQ = all.quantile(0.25)
    UQ = all.quantile(0.75)
    pvalue1 = mannwhitneyu(dist['IMD'].to_numpy(), all.tolist(),
                           alternative='greater', nan_policy='omit')[1]
    pvalue2 = mannwhitneyu(dist['IMD'].to_numpy(), all.tolist(),
                           alternative='less', nan_policy='omit')[1]
    axis.axvline(median, color='red', linestyle='dashed')
    axis.annotate(f' Median: {median:.0f}',
                  xy=(median, dist['Proportion'].max()-0.005),
                  xytext= (median, dist['Proportion'].max()), color='red',
                  fontsize='medium')
    axis.axvline(LQ, color='grey', linestyle='dashed')
    axis.annotate(f' LQ: {LQ:.0f}', xy=(LQ, dist['Proportion'].max()-0.005),
                  xytext=(LQ, dist['Proportion'].max()), color='grey',
                  fontsize='medium')
    axis.axvline(UQ, color='grey', linestyle='dashed')
    axis.annotate(f' UQ: {UQ:.0f}', xy=(UQ, dist['Proportion'].max()-0.005),
                  xytext=(UQ, dist['Proportion'].max()), color='grey',
                  fontsize='medium')
    if pvalue1 < 0.05:
        axis.annotate('IMD Distribution\nlower than\ngeneral population\np-value={pvalue1:.2e}',
                      xy=(8.5, dist['Proportion'].max()*0.75),
                      xytext=(8.5, dist['Proportion'].max()*0.75), color='black',
                      fontsize='medium')
    if pvalue2 < 0.05:
        axis.annotate('IMD Distribution\ngreater than\nPlymouth population\np-value={pvalue2:.2e}',
                      xy=(8.5, dist['Proportion'].max()*0.75),
                      xytext=(8.5, dist['Proportion'].max()*0.75), color='black',
                      fontsize='medium')
    #Set subtitle and xticks
    axis.set_title(title, fontsize='large', fontweight='bold')
    axis.set_xticks(IMDs)
fig.suptitle('Proportion of the Population within each IMD',
             fontsize='xx-large', fontweight='bold')
fig.set_figheight(15)
fig.set_figwidth(25)
plt.savefig('Plots/IMD Proportions.png')
plt.close()

###################Proportion of comm/hosp by IMD plot####################
comm_IMD_dist = IMD_dist(comm_deaths.loc[comm_deaths['Date of Death'] > pd.to_datetime('01-01-2023')].copy(), 'NHS Number')
hosp_IMD_dist = IMD_dist(hosp_deaths.loc[hosp_deaths['Date of Death'] > pd.to_datetime('01-01-2023')].copy(), 'Hospital Number')
IMD_prop = (comm_IMD_dist[['IMD', 'Count']]
            .merge(hosp_IMD_dist[['IMD', 'Count']], on='IMD',
                   suffixes=[' Comm', ' Hosp']))
IMD_prop['Total'] = IMD_prop[['Count Comm', 'Count Hosp']].sum(axis=1)
IMD_prop['Community'] = IMD_prop['Count Comm'] / IMD_prop['Total']
IMD_prop['Hospital'] = IMD_prop['Count Hosp'] / IMD_prop['Total']
pvalue = mannwhitneyu(comm_deaths.loc[comm_deaths['Date of Death'] > pd.to_datetime('01-01-2023'), 'IMD'],
                      hosp_deaths.loc[hosp_deaths['Date of Death'] > pd.to_datetime('01-01-2023'), 'IMD'],
                        alternative='greater', nan_policy='omit')[1]
#####Plot
# Figure setup
plt.figure(figsize=(10, 15))
ys = range(len(IMD_prop))[::-1]
height = 0.8
base = 0
# Draw bars for Community
for y, value in zip(ys, IMD_prop['Community'].values*100):
    plt.broken_barh([(base, -np.abs(base-value))], (y - height/2,height),
                    facecolors=['#0d47a1','#0d47a1'], label='Community')
# Draw bars for Hospital
for y, value2 in zip(ys, IMD_prop['Hospital'].values*100):
    plt.broken_barh([(base, np.abs(base-value2))], (y - height/2, height),
                    facecolors=['#e2711d','#e2711d'], label='Hospital')
if pvalue < 0.05:
    plt.text(-70, 7, f'Community deaths have\na higher IMD than\nHospital deaths\np-value={pvalue:.2e}',
             fontsize=12, horizontalalignment='center',
             verticalalignment='center')
# Modify the graph
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.yticks(ys, IMD_prop['IMD'].values.tolist())
plt.xticks(np.linspace(-100, 100, 21), [abs(int(i)) for i in np.linspace(-100, 100, 21)])
plt.xlabel('% of IMD deaths')
plt.ylabel('IMD')
plt.grid(linewidth=0.1, color='black')
plt.title('Percentage Split of Deaths by IMD (since 2023)')
plt.rcParams["figure.figsize"] = (20,10)
plt.savefig('Plots/Percentage Split of Deaths by IMD.png')
plt.close()
############################Age of Death comparison#############################
def split(deaths, col, values, labels):
    age_dist = (deaths.groupby([col, 'Age of Death'], as_index=False)
                 ['Date of Death'].count())
    N = (age_dist.loc[age_dist[col] == values[0]]
        .merge(pd.DataFrame({'Age of Death' : all_ages}), on='Age of Death',
        how='outer').sort_values(by='Age of Death')['Date of Death'].fillna(0)
        .values)             
    Y = (age_dist.loc[age_dist[col] == values[1]]
        .merge(pd.DataFrame({'Age of Death' : all_ages}), on='Age of Death',
        how='outer').sort_values(by='Age of Death')['Date of Death'].fillna(0)
        .values)
    age_dist = {labels[0]:N, labels[1]:Y}
    return age_dist
hosp_age_dist = split(hosp_deaths, 'ED Death', ['N','Y'], ['Other Ward', 'ED'])
comm_age_dist = split(hosp_deaths, 'carehome', ['N','Y'], ['Non Carehome', 'Carehome'])

fig, axs = plt.subplots(2)
#Loop through and plot each group as a subplot.
subplots = [(axs[0], hosp_age_dist, hosp_deaths['Age of Death'],
             hosp_deaths['ED Death'] == 'Y', 'Hospital Deaths'),
            (axs[1], comm_age_dist, comm_deaths['Age of Death'],
             comm_deaths['carehome'] == 'Y', 'Community Deaths')]
for axis, dist, all, mask, title in subplots:
    #bar chart
    y_lab_pos = []
    whitneyu_inputs = []
    for label, values in dist.items():
        axis.bar(all_ages, values, label=label)
        y_lab_pos.append(values.max())
        whitneyu_inputs.append(values)
    y_lab_pos = max(y_lab_pos)
    axis.legend(loc='upper right')
    #Whitney u tests
    keys = list(dist.keys())
    for test in ['greater', 'less']:
        #Subsets against each other
        pvalue1 = mannwhitneyu(all[~mask], all[mask],
                              alternative=test, nan_policy='omit')[1]
        #location by overall population
        pvalue2 = mannwhitneyu(all, population['Age of Death'],
                               alternative=test, nan_policy='omit')[1]
        test = 'lesser' if test == 'less' else test
        if pvalue1 < 0.05:
            text1 = f'{keys[0]} has a {test}\nAge of Death than {keys[1]}\np-value = {pvalue1:.2e}'
            axis.text(20, 200, text1, fontsize='medium')
        if pvalue2 < 0.05:
            text2 = f'{title} have a {test}\nAge of Death than the\nwhole deaths population\np-value = {pvalue2:.2e}'
            axis.text(20, 100, text2, fontsize='medium')
    #LQ, median and UQ lines
    median = all.median()
    LQ = all.quantile(0.25)
    UQ = all.quantile(0.75)
    #Add mean, LQ and UQ lines
    axis.axvline(median, color='red', linestyle='dashed')
    axis.annotate(f' Median: {median:.0f}',
                  xy=(median, y_lab_pos-0.005),
                  xytext= (median, y_lab_pos), color='red',
                  fontsize='medium')
    axis.axvline(LQ, color='grey', linestyle='dashed')
    axis.annotate(f' LQ: {LQ:.0f}', xy=(LQ, y_lab_pos-0.005),
                  xytext=(LQ, y_lab_pos), color='grey',
                  fontsize='medium')
    axis.axvline(UQ, color='grey', linestyle='dashed')
    axis.annotate(f' UQ: {UQ:.0f}', xy=(UQ, y_lab_pos-0.005),
                  xytext=(UQ, y_lab_pos), color='grey',
                  fontsize='medium')
    #Set subtitle and xticks
    axis.set_title(title, fontsize='large', fontweight='bold')
fig.suptitle('Age of Death by Location',
             fontsize='xx-large', fontweight='bold')
fig.set_figheight(15)
fig.set_figwidth(25)
plt.savefig('Plots/Age of Death by Location.png')
plt.close()
##############################Age of Death by IMD###############################
input_data = [population.loc[population['IMD'] == i, 'Age of Death'].dropna() for i in range(1, 11)]
values = [(int(i.min()), int(i.median()), int(i.max()), int(i.count())) for i in input_data]
plt.violinplot(input_data, showmedians=True)
overall_max = max([i[2] for i in values])
#Get all pairs of decile numbers with no repeats and hypothesis test
imd_pvals = []
pair_list = itertools.combinations(range(0,10),2)
for r in pair_list:
    for test in ['greater', 'less']:
        #Subsets against each other
        pvalue = mannwhitneyu(input_data[r[0]], input_data[r[1]],
                              alternative=test, nan_policy='omit')[1]
        test = 'lesser' if test=='less' else test
        #Record statistically significant results
        if pvalue < 0.05:
            imd_pvals.append([pvalue, r[0]+1, r[1]+1, test])
#Create output text for IMD tests.
imd_prop_test = pd.DataFrame(imd_pvals, columns=['pvalue', 'IMD1', 'IMD2', 'test'])
pairs = (imd_prop_test.groupby(['IMD1','test'], as_index=False)['IMD2']
         .apply(list).values.tolist())
text = ''
i=0
for pair in pairs:
    imd = pair[0]
    test = pair[1]
    lst = pair[2]
    seccond_str = ('all other IMD values' if len(lst) == (10 - imd)
                    else ('IMDs ' +   ', '.join(map(str, lst))))
    string = f'IMD {imd} {test} than {seccond_str},\n'
    text += string

for i, (min_, med, max_, count_) in enumerate(values):
    plt.text(i+1, min_+2, str(min_), fontsize=10)
    plt.text(i+1, med, str(med), fontsize=10)
    plt.text(i+1, (max_-4), str(max_), fontsize=10)
    plt.text(i+0.9, (overall_max+2), str(count_), fontsize=10)
plt.text(-0.75, values[0][0], 'Min:', fontsize=12, fontweight='bold')
plt.text(-0.75, values[0][1], 'Median:', fontsize=12, fontweight='bold')
plt.text(-0.75, values[0][2], 'Max:', fontsize=12, fontweight='bold')
plt.text(-0.75, overall_max + 1, 'Total:', fontsize=12, fontweight='bold')
plt.text(1, 10, text, fontsize=12)
plt.xticks([i for i in range(1,11)])
plt.xlabel('IMD')
plt.ylabel('Age of Death')
plt.title('Age of Death by IMD', fontweight='bold', fontsize='xx-large')
plt.savefig('Plots/Age of Deah by IMD.png')
plt.close()

########################Age of Death by IMD in Comm/Hosp########################
hosp_input_data = [hosp_deaths.loc[hosp_deaths['IMD'] == i, 'Age of Death'].dropna() for i in range(1, 11)]
comm_input_data = [comm_deaths.loc[comm_deaths['IMD'] == i, 'Age of Death'].dropna() for i in range(1, 11)]
fig, axs = plt.subplots(2, sharex=True)
inputs = [(axs[0], hosp_input_data, 'Hospital Age of Death'),
          (axs[1], comm_input_data, 'Community Age of Death')]
for ax, data, title in inputs:
    values = [(int(i.min()), int(i.median()), int(i.max()), int(i.count())) for i in data]
    ax.violinplot(data, showmedians=True)
    #Get all pairs of decile numbers with no repeats and hypothesis test
    imd_pvals = []
    pair_list = itertools.combinations(range(0,10),2)
    for r in pair_list:
        for test in ['greater', 'less']:
            #Subsets against each other
            pvalue = mannwhitneyu(input_data[r[0]], input_data[r[1]],
                                alternative=test, nan_policy='omit')[1]
            test = 'lesser' if test=='less' else test
            #Record statistically significant results
            if pvalue < 0.05:
                imd_pvals.append([pvalue, r[0]+1, r[1]+1, test])
    #create output text for hypothesis tests
    imd_prop_test = pd.DataFrame(imd_pvals, columns=['pvalue', 'IMD1', 'IMD2', 'test'])
    pairs = (imd_prop_test.groupby(['IMD1','test'], as_index=False)['IMD2']
            .apply(list).values.tolist())
    text = ''
    i=0
    for pair in pairs:
        imd = pair[0]
        test = pair[1]
        lst = pair[2]
        seccond_str = ('all other IMD values' if len(lst) == (10 - imd)
                        else ('IMDs ' +   ', '.join(map(str, lst))))
        string = f'IMD {imd} {test} than {seccond_str},\n'
        text += string

    overall_max = max([i[2] for i in values])
    for i, (min_, med, max_, count_) in enumerate(values):
        ax.text(i+1, min_+2, str(min_), fontsize=10)
        ax.text(i+1, med, str(med), fontsize=10)
        ax.text(i+1, (max_-5), str(max_), fontsize=10)
        ax.text(i+0.9, (overall_max+1), str(count_), fontsize=10)
    ax.text(-0.75, values[0][0], 'Min:', fontsize=12, fontweight='bold')
    ax.text(-0.75, values[0][1], 'Median:', fontsize=12, fontweight='bold')
    ax.text(-0.75, values[0][2]-2, 'Max:', fontsize=12, fontweight='bold')
    ax.text(-0.75, overall_max+1, 'Total:', fontsize=12, fontweight='bold')
    ax.text(10.2, 10, text, fontsize=12)
    ax.set_ylabel('Age of Death')
    ax.set_title(title, fontsize='large', fontweight='bold')
plt.xticks([i for i in range(1,11)])
plt.xlabel('IMD')
plt.savefig('Plots/Age of Deah by IMD on Hospital and Community.png', bbox_inches='tight')
plt.close()

##########################Age of Death by bianry col############################
def splits(dist, split_col, split_variables, split_labels):
    data_var0 = (dist.loc[dist[split_col] == split_variables[0]]
                 .merge(pd.DataFrame({'Age of Death':all_ages}),
                 on='Age of Death', how='outer').sort_values(by='Age of Death')
                 ['Date of Death'].fillna(0).values)
    data_var1 = (dist.loc[dist[split_col] == split_variables[1]]
                 .merge(pd.DataFrame({'Age of Death':all_ages}),
                 on='Age of Death', how='outer').sort_values(by='Age of Death')
                 ['Date of Death'].fillna(0).values)
    splits = {split_labels[0]:data_var0, split_labels[1]:data_var1}
    return splits


def plot_age_location_by_variable(hosp_deaths, comm_deaths, extra_variable,
                                  variable_options, variable_names, plt_title):
    #####Hospital
    hosp_dist = (hosp_deaths.groupby([extra_variable, 'ED Death', 'Age of Death'],
                                            as_index=False)
                                            ['Date of Death'].count())
    #Variable options by ED and other wards
    hosp_1 = hosp_dist.loc[hosp_dist[extra_variable] == variable_options[0]].copy()
    hosp_1 = splits(hosp_1, 'ED Death', ['N', 'Y'], ['Other Ward', 'ED'])
    all_hosp_1 = hosp_deaths.loc[hosp_deaths[extra_variable] == variable_options[0]].copy()
    #Variable option 2 by ED and other wards
    hosp_2 = hosp_dist.loc[hosp_dist[extra_variable] == variable_options[1]].copy()
    hosp_2 = splits(hosp_2, 'ED Death', ['N', 'Y'], ['Other Ward', 'ED'])
    all_hosp_2 = hosp_deaths.loc[hosp_deaths[extra_variable] == variable_options[1]].copy()
    #####Community
    comm_dist = (comm_deaths.groupby([extra_variable, 'carehome', 'Age of Death'],
                                            as_index=False)
                                            ['Date of Death'].count())
    #Variable options by ED and other wards
    comm_1 = comm_dist.loc[comm_dist[extra_variable] == variable_options[0]].copy()
    comm_1 = splits(comm_1, 'carehome', ['N', 'Y'], ['Non Carehome', 'Carehome'])
    all_comm_1 = comm_deaths.loc[comm_deaths[extra_variable] == variable_options[0]].copy()
    #Variable option 2 by ED and other wards
    comm_2 = comm_dist.loc[comm_dist[extra_variable] == variable_options[1]].copy()
    comm_2 = splits(comm_2, 'carehome', ['N', 'Y'], ['Non Carehome', 'Carehome'])
    all_comm_2 = comm_deaths.loc[comm_deaths[extra_variable] == variable_options[1]].copy()

    #######Plot
    fig, axs = plt.subplots(2, 2, sharex=True)
    #Loop through and plot each group as a subplot.
    subplots = [(axs[0, 0], hosp_1, all_hosp_1['Age of Death'],
                 all_hosp_1['ED Death'] == 'Y',
                 f'{variable_names[0]} Hospital Deaths'),
                (axs[0, 1], comm_1, all_comm_1['Age of Death'],
                all_comm_1['carehome'] == 'Y',
                f'{variable_names[0]} Community Deaths'),
                (axs[1, 0], hosp_2, all_hosp_2['Age of Death'],
                all_hosp_2['ED Death'] == 'Y',
                f'{variable_names[1]} Hospital Deaths'),
                (axs[1, 1], comm_2, all_comm_2['Age of Death'],
                all_comm_2['carehome'] == 'Y',
                f'{variable_names[1]} Community Deaths')]

    for axis, dist, all, mask, title in subplots:
        #bar chart
        y_lab_pos = []
        for label, values in dist.items():
            axis.bar(all_ages, values, label=label)
            y_lab_pos.append(values.max())
        y_lab_pos = max(y_lab_pos)
        #Whitney u tests
        keys = list(dist.keys())
        for test in ['greater', 'less']:
            #Subsets against each other
            mask_all = mask.sum()
            if not ((mask_all == len(all)) or (mask_all == 0)):
                pvalue1 = mannwhitneyu(all[~mask], all[mask],
                                    alternative=test, nan_policy='omit')[1]
                #location by overall population
                pvalue2 = mannwhitneyu(all, population['Age of Death'],
                                    alternative=test, nan_policy='omit')[1]
                test = 'lesser' if test == 'less' else test
                if pvalue1 < 0.05:
                    text1 = f'{keys[0]} has a {test}\nAge of Death than {keys[1]}\np-value = {pvalue1:.2e}'
                    axis.text(0, y_lab_pos*0.6, text1, fontsize='large')
                if pvalue2 < 0.05:
                    text2 = f'{title}\nhave a {test} Age of Death\nthan the whole deaths population\np-value = {pvalue2:.2e}'
                    axis.text(0, y_lab_pos*0.3, text2, fontsize='large')
                axis.legend(loc='upper right')
        #LQ, median and UQ lines
        median = all.median()
        LQ = all.quantile(0.25)
        UQ = all.quantile(0.75)
        axis.axvline(median, color='red', linestyle='dashed')
        axis.annotate(f' Median: {median:.0f}', xy=(median, y_lab_pos),
                      xytext=(median, y_lab_pos), color='red', fontsize='x-large')
        axis.axvline(LQ, color='grey', linestyle='dashed')
        axis.annotate(f' LQ: {LQ:.0f}', xy=(LQ, y_lab_pos*0.9),
                      xytext=(LQ, y_lab_pos*0.9), color='grey', fontsize='x-large')
        axis.axvline(UQ, color='grey', linestyle='dashed')
        axis.annotate(f' UQ: {UQ:.0f}', xy=(UQ, y_lab_pos*0.9),
                      xytext=(UQ, y_lab_pos*0.9), color='grey', fontsize='x-large')
        #Set subtitle and xticks
        axis.set_title(title, fontsize='xx-large', fontweight='bold')
    for ax in axs.flat:
        ax.set(xlabel='Age of Death')
    fig.suptitle(plt_title,
                fontsize='xx-large', fontweight='bold')
    fig.set_figheight(20)
    fig.set_figwidth(35)
    plt.savefig(f'Plots/{plt_title}.png')
    plt.close()

#Age of Death by location by sex
hosp_deaths['carehome copy'] = hosp_deaths['carehome'].copy()
comm_deaths['carehome copy'] = comm_deaths['carehome'].copy()
plot_age_location_by_variable(hosp_deaths, comm_deaths, 'carehome copy', ['Y', 'N'],
                              ['Carehome Resident', 'Non Carehome Resident'],
                              'Age of Death by Carehome Residency and Location')
#Age of Death by location by sex
plot_age_location_by_variable(hosp_deaths, comm_deaths, 'sexxx', ['1', '2'],
                              ['Male', 'Female'],
                              'Age of Death by Sex and Location')
#Age of Death by location by ethnicity
plot_age_location_by_variable(hosp_deaths, comm_deaths, 'Ethnicity',
                              ['White British', 'Ethnic Minority'],
                              ['White British', 'Ethnic Minority'],
                              'Age of Death by Ethnicity and Location')

###############################Deaths per Month#################################
def month_of_death(df, date_col, name_col):
    df['Month of Death'] = df[date_col].dt.to_period('M').dt.to_timestamp()
    group = (df.groupby('Month of Death', as_index=False)[name_col].count()
             .rename(columns={name_col:'Count'}).sort_values(by='Month of Death'))
    return group
comm_CH_deaths_by_date = month_of_death(comm_CH_deaths, 'Date of Death', 'First Name')
comm_oth_deaths_by_date = month_of_death(comm_oth_deaths, 'Date of Death', 'First Name')
hosp_ED_deaths_by_date = month_of_death(hosp_ED_deaths, 'Date of Death', 'First Name')
hosp_oth_deaths_by_date = month_of_death(hosp_oth_deaths, 'Date of Death', 'First Name')
baby_deaths_by_date = month_of_death(baby_deaths, '16+5', 'Mothers Forename')
#####Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
fig.suptitle('Number of Deaths per Month')
#Hospital
ax1.plot(hosp_ED_deaths_by_date['Month of Death'], hosp_ED_deaths_by_date['Count'],
         label='ED')
ax1.plot(hosp_oth_deaths_by_date['Month of Death'], hosp_oth_deaths_by_date['Count'],
         label='Other Ward')
ax1.legend()
ax1.set_title('Hospital Deaths')
#Community
ax2.plot(comm_CH_deaths_by_date['Month of Death'], comm_CH_deaths_by_date['Count'],
         label='Carehome')
ax2.plot(comm_oth_deaths_by_date['Month of Death'], comm_oth_deaths_by_date['Count'],
         label='Non Carehome')
ax2.legend()
ax2.set_title('Community Deaths')
#Baby
ax3.set_title('Hospital and Community Deaths')
ax3.plot(baby_deaths_by_date['Month of Death'], baby_deaths_by_date['Count'])
ax3.set_title('Baby Deaths')
plt.savefig('Plots/Number of Deaths per Month.png')

##################################Deaths by DoW#################################
#######Analysis
def deaths_by_dow(df, count_col):
    df['Day of Death'] = pd.Categorical(df['Day of Death'],
                         ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    group = (df.groupby('Day of Death', as_index=False)[count_col].count()
             .rename(columns={count_col:'Count'}))
    if len(group) > 7:
        group = group.sort_values(by='Count').tail(7)
    #Statistical significance test
    chi = chisquare(f_obs=group['Count'],
                    f_exp=[1/7 * group['Count'].sum()]*7)
    pvalue = chi[1]
    if pvalue < 0.05:
        text = f'Significant Difference\nin Deaths by Day of Week\np-value={pvalue:.2e}'
    else:
        text = f''
    return group, text

baby_deaths['Day of Death'] = baby_deaths['16+5'].dt.day_name().str[:3]
(hosp_ED_death_dow, hosp_ED_death_text) = deaths_by_dow(hosp_ED_deaths, 'First Name')
(hosp_oth_death_dow, hosp_oth_death_text) = deaths_by_dow(hosp_oth_deaths, 'First Name')
(comm_CH_death_dow, comm_CH_death_text) = deaths_by_dow(comm_CH_deaths, 'First Name')
(comm_oth_death_dow, comm_oth_death_text) = deaths_by_dow(comm_oth_deaths, 'First Name')
#######Plot
fig, axs = plt.subplots(2, 2, sharex=True)
#Loop through and plot each group as a subplot.
subplots = [(axs[0, 0], hosp_ED_death_dow, hosp_ED_death_text, 'Hospital ED Deaths'),
            (axs[1, 0], hosp_oth_death_dow, hosp_oth_death_text, 'Hospital Other Ward Deaths'),
            (axs[0, 1], comm_CH_death_dow, comm_CH_death_text, 'Community Carehome Deaths'),
            (axs[1, 1], comm_oth_death_dow, comm_oth_death_text, 'Community Non Carehome Deaths')]

for axis, dist, text, title in subplots:
    #bar chart
    axis.bar(dist['Day of Death'].values, dist['Count'].values,
             color='cornflowerblue')
    axis.text(3, dist['Count'].mean()*0.5, text, horizontalalignment='center')
    #Set subtitle and xticks
    axis.set_title(title, fontsize='large', fontweight='bold')
fig.suptitle('Deaths by Day of Week',
             fontsize='xx-large', fontweight='bold')
fig.set_figheight(10)
fig.set_figwidth(15)
plt.savefig('Plots/Deaths by Day of Week.png')
plt.close()

##################################Deaths by HoD#################################
#######Analysis
def deaths_by_hod(df):
    hours = (pd.to_numeric(df['Time of death'].dropna().astype(str).str[:2],
                           errors='coerce').dropna())
    hours = hours.value_counts().reset_index()
    hours.columns = ['Hour of Day', 'Count']
    hours = hours.sort_values(by='Hour of Day')
        #Statistical significance test
    chi = chisquare(f_obs=hours['Count'],
                    f_exp=[1/24 * hours['Count'].sum()]*24)
    pvalue = chi[1]
    if pvalue < 0.05:
        text = f'Significant Difference\nin Deaths by hour of the day\np-value={pvalue:.2e}'
    else:
        text = f''
    return hours, text

(hosp_ED_death_hod, hosp_ED_death_text) = deaths_by_hod(hosp_ED_deaths)
(hosp_oth_death_hod, hosp_oth_death_text) = deaths_by_hod(hosp_oth_deaths)
(comm_CH_death_hod, comm_CH_death_text) = deaths_by_hod(comm_CH_deaths)
(comm_oth_death_hod, comm_oth_death_text) = deaths_by_hod(comm_oth_deaths)

#######Plot
fig, axs = plt.subplots(2, 2, sharex=True)
#Loop through and plot each group as a subplot.
subplots = [(axs[0, 0], hosp_ED_death_hod, hosp_ED_death_text, 'Hospital ED Deaths'),
            (axs[1, 0], hosp_oth_death_hod, hosp_oth_death_text, 'Hospital Other Ward Deaths'),
            (axs[0, 1], comm_CH_death_hod, comm_CH_death_text, 'Community Carehome Deaths'),
            (axs[1, 1], comm_oth_death_hod, comm_oth_death_text, 'Community Non Carehome Deaths')]

for axis, dist, text, title in subplots:
    #bar chart
    axis.bar(dist['Hour of Day'].values, dist['Count'].values,
             color='cornflowerblue')
    axis.text(11, dist['Count'].mean()*0.5, text, horizontalalignment='center')
    #Set subtitle and xticks
    axis.set_title(title, fontsize='large', fontweight='bold')
fig.suptitle('Deaths by Hour of Day',
             fontsize='xx-large', fontweight='bold')
for ax in axs.flat:
    ax.set(xlabel='Hour of Day')
fig.set_figheight(10)
fig.set_figwidth(15)
plt.savefig('Plots/Deaths by Hour of Day.png')
plt.close()

#################################Causes of Death################################
population.groupby(['IMD', 'Cause of death'], as_index=False)['First Name'].count()
cause_of_death = (pd.DataFrame(hosp_deaths['Cause of death'].str.lower().value_counts())
                  .join(comm_deaths['Cause of death'].str.lower().value_counts(),
                        lsuffix=' Hosp', rsuffix=' Comm')).fillna(0)
top_idx = cause_of_death.sort_values(by='count Hosp').tail(6).index.append(
          cause_of_death.sort_values(by='count Comm').tail(6).index)
cause_of_death = cause_of_death.loc[top_idx].drop_duplicates().copy()
cause_of_death.columns = ['Hospital', 'Community']
cause_of_death.plot(kind='bar', title='Most Common Causes of Death',
                    figsize=(25,15), rot=20)
plt.savefig('Plots/Causes of Death.png')
plt.close()

############################Causes of Death Heatmap#############################
#Make causes lower case to combine more into one
population['Cause of death'] = population['Cause of death'].str.lower().str.strip()
#Get the normalised cause of death as a proportion of the whole population
# #for that IMD
IMD_total = population.groupby('IMD', as_index=False)['Surname'].count()
cod_IMD = population.groupby(['IMD', 'Cause of death'],
                             as_index=False)['First Name'].count()
cod_IMD = cod_IMD.merge(IMD_total, on='IMD')
cod_IMD['Normalised Proportion'] = cod_IMD['First Name'] / cod_IMD['Surname']
cod_IMD['IMD'] = cod_IMD['IMD'].astype(int)
cod_IMD_table = cod_IMD.pivot(index='Cause of death', columns='IMD',
                              values='Normalised Proportion')
#Filter to rows whith a higher value
cod_IMD_table = cod_IMD_table.loc[(cod_IMD_table.max(axis=1)>0.005)].copy()
#plot
fig, ax = plt.subplots(figsize=(25, 15))
sns.heatmap(cod_IMD_table, cmap='Blues', robust=True, annot=True, fmt='.2f',
            linewidths=0.5, linecolor='k',
            xticklabels=cod_IMD_table.columns, ax=ax)
ax.set(xlabel='IMD', ylabel='Cause of Death')
plt.title('Cause of Death by IMD\nNormalised to the Proportion of Total Deaths in each IMD')
plt.savefig('Plots/Cause of Death by IMD.png', bbox_inches='tight')
plt.close()