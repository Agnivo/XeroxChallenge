import pandas as pa
import numpy as np
import math


def generateFeatures():
    vitals = pa.read_csv(
        'Training_Dataset/id_time_vitals_train.csv',
        dtype={'ID': np.int32, 'TIME': np.int32, 'ICU': np.int32}
    )
    labs = pa.read_csv(
        'Training_Dataset/id_time_labs_train.csv',
        dtype={'ID': np.int32, 'TIME': np.int32}
    )

    ids = np.asarray(np.unique(np.asarray(vitals['ID'])))

    features = {}
    vitalColumns = [str(vitals.columns[i])
                    for i in xrange(vitals.columns.size)]

    for i, row in enumerate(vitals.iterrows()):
        row[1]['TIME'] = int(row[1]['TIME'] / 3600)
        if row[1]['ID'] not in features.keys():
            features[row[1]['ID']] = {}
        features[row['ID']][row[1]['TIME']] 


    labColumns = [str(labs.columns[i])
                  for i in xrange(labs.columns.size)]

    for i, row in enumerate(vitals.iterrows()):
        row[1]['TIME'] = int(row[1]['TIME'] / 3600)
