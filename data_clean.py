import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

input_data = "data/data.csv"

df_all = pd.read_csv(input_data)

train_raw, test_raw = train_test_split(df_all, test_size=0.25)


#This takes a panda dataframe and cleans the medians
#Make sure to put test data and training data in separately 
def process_data(df) :
	for name in df.columns :
		lastTwo = df[name]%100
		df[name][lastTwo == 99] = 0 #Legitimate skips are 0
		df[name][lastTwo == 93] = 0 #If no drug has been used in period of interest, we give them 0
		df[name][lastTwo == 91] = 0 #No drug ever used gets 0
		df[name][lastTwo == 81] = 0 #Logically assigned no drug gets 0
		df[name][lastTwo == 83] = 0 #Logically assigned no drug used in period of interest
		df[name][lastTwo == 89] = 0 #Logically assigned
		median = np.median(df[name])
		df[name][lastTwo == 94] = median #Don't know
		df[name][lastTwo == 97] = median #Refused
		df[name][lastTwo == 98] = median #Blank
		df[name][lastTwo == 85] = median #Bad data
		if sum(df[name] > 9) > 0 :
			print name

	zero_ones = ["CIGEVER", "SMKLSSEVR", "CIGAREVR", "ALCEVER",
					"MJEVER", "HEREVER",
					'LSD',
	              'PCP',
	              'PEYOTE',
	              'MESC',
	              'PSILCY',
	              'ECSTMOLLY',
	              'KETMINESK',
	              'DMTAMTFXY',
	              'SALVIADIV',
	              'HALLUCOTH',
	              'AMYLNIT',
	              'CLEFLU',
	              'GAS',
	              'GLUE',
	              'ETHER',
	              'SOLVENT',
	              'LGAS',
	              'NITOXID',
	              'FELTMARKR',
	              'SPPAINT',
	              'AIRDUSTER',
	              'OTHAEROS',
	              'INHALEVER', 
	              "METHAMEVR",
	              "OXCNANYYR",
	              "TRQANYLIF",
	              "STMANYLIF",
	              "SEDANYLIF",
	              "PNRNMLIF"]

	for name in zero_ones :
		print name
		lastTwo = df[name]%100
		df[name][lastTwo == 2] = 0 #Nos are encoded as 2s, but we want them to be 0.

	return df

train_clean = process_data(train_raw)
test_clean = process_data(test_raw)

train_clean.to_csv("training_clean.csv", index = False)
test_clean.to_csv("testing_clean.csv", index = False)
