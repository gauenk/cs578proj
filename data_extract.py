import csv

input_tsv = 'NSDUH-2015-DS0001-bndl-data-tsv/NSDUH-2015-DS0001-data/NSDUH-2015-DS0001-data-excel.tsv'
#input_tsv = 'NSDUH-2015-DS0001-bndl-data-tsv/NSDUH-2015-DS0001-data/short.tsv'
output_csv = 'data/data.csv'

# CIGREC - has some unordered people with the logically assigned
# Find max value of ALCUS30D

# Want to predict COCEVER + CRKEVER - convert into one column
# COCCRK

# Throw out rows with responses which start with 9 (or replace with average response)
# Except 91, 93
# Note to future self: For the "ever" questions, convert responses not in 1, 2 to 0.5 for refused/don't know answers
fieldnames = ['CIGEVER', 'CIGOFRSM', 'CIGWILYR', 'CIGTRY', 'CIGREC', 'CIG30AV',
              'SMKLSSEVR', 'SMKLSSTRY', 'SMKLSSREC',
              'CIGAREVR', 'CIGARTRY', 'CIGARREC', 'CGR30USE',
              'ALCEVER', 'ALCTRY', 'ALCREC', 'ALDAYPYR', 'ALCUS30D', 'ALCBNG30D',
              'MJEVER', 'MJAGE', 'MJREC', 'MJYRTOT',

              'HEREVER', 'HERAGE', 'HERREC', 'HERYRTOT', 'HER30USE',
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
              'METHAMEVR', 'METHAMAGE', 'METHAMREC', 'METHDYSYR',
              'OXCNANYYR', 'PNRANYLIF', 'PNRANYREC',
              'TRQANYLIF', 'TRQANYREC',
              'STMANYLIF', 'STMANYREC',
              'SEDANYLIF', 'SEDANYREC',
              'PNRNMLIF', 'OXCNNMYR',

              # Note, do we want the imputed data?
              'IRCIGRC',
              'IRCGRRC',
              'IRPIPLF',
              'IRALCRC',
              'IRMJRC',
              'IRHERRC',

              # Criminal record
              'BOOKED',

              # Demographic data
              'SERVICE',
              'HEALTH',
              'MOVSINPYR2',
              'SEXATRACT',
              'SPEAKENGL',
              'DIFFTHINK',
              'IRSEX',
              'IRMARITSTAT',
              'IREDUHIGHST2',
              'CATAGE',
              'EDUENROLL',
              'WRKDHRSWK2',
              'WRKDRGPOL',
              'WRKTSTDRG',
              'MEDICARE',
              'CAIDCHIP',
              'PRVHLTIN',
              'HLTINMNT',
             ]

data = []

print('Loading...')
with open(input_tsv, 'r') as in_file:
    reader = csv.DictReader(in_file, delimiter='\t')
    
    for row in reader:
        row_data = {}
        for field in fieldnames:
            row_data[field] = row[field]

        # Combine crack and cocaine usage into one variable
        did_cocaine = row['COCEVER']
        did_crack = row['CRKEVER']
        did_crack_cocaine = '1' if did_cocaine == '1' or did_crack == '1' else '2'
        row_data['CRKCOC'] = did_crack_cocaine

        data.append([row_data[x] for x in fieldnames + ['CRKCOC']])

print('Saving...')
with open(output_csv, 'w') as out_file:
    writer = csv.writer(out_file)
    writer.writerow(fieldnames + ['CRKCOC'])

    for entry in data:
        writer.writerow(entry)


