import pandas as pd
import numpy as n
import sys

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

for i in range(1, len(sys.argv)):
    feature = int(sys.argv[i])
    print(fieldnames[feature-1])