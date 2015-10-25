import sys
import pandas as pd
import math


def main(filein, fileout):
    train_vitals = pd.read_csv(filein)
    min_vitals = {}
    max_vitals = {}

    min_vitals['V1'] = 80.0
    max_vitals['V1'] = 120.0

    min_vitals['V2'] = 60.0
    max_vitals['V2'] = 80.0

    min_vitals['V3'] = 60.0
    max_vitals['V3'] = 100.0

    min_vitals['V4'] = 10.0
    max_vitals['V4'] = 20.0

    min_vitals['V5'] = 90.0
    max_vitals['V5'] = 100.0

    min_vitals['V6'] = 95.0
    max_vitals['V6'] = 100.0

    key_vitals = train_vitals.keys()
    num = len(train_vitals[key_vitals[0]])

    maxval = {}
    nacount = 0

    scale_feats = {}
    for key in key_vitals:
        scale_feats[key] = []

    for i in xrange(0, num):
        scale_feats[key_vitals[0]].append(train_vitals[key_vitals[0]][i])
        scale_feats[key_vitals[1]].append(train_vitals[key_vitals[1]][i])
        scale_feats[key_vitals[-1]].append(train_vitals[key_vitals[-1]][i])
        for key in key_vitals[2:-1]:
            if not math.isnan(train_vitals[key][i]):
                if key not in maxval:
                    maxval[key] = train_vitals[key][i]
                else:
                    maxval[key] = max(maxval[key], train_vitals[key][i])

                scale_feats[key].append(
                    (train_vitals[key][i] - min_vitals[key]) /
                    (max_vitals[key] - min_vitals[key])
                )
                if scale_feats[key][i] > 10.0:
                    nacount += 1
            else:
                scale_feats[key].append(train_vitals[key][i])

    print maxval
    print nacount

    with open(fileout, 'w') as f:
        f.write(key_vitals[0])
        for key in key_vitals[1:]:
            f.write(',' + key)
        f.write('\n')

        for i in xrange(0, num):
            f.write(
                str(scale_feats[key_vitals[0]][i]) + ','
                + str(scale_feats[key_vitals[1]][i])
            )
            for key in key_vitals[2:]:
                if math.isnan(scale_feats[key][i]):
                    f.write(',NA')
                else:
                    if scale_feats[key][i] > 10.0 or scale_feats[key][i] < -10.0:
                        # replace by f.write(',str(scale_feats[key][i])')
                        f.write(',NA')
                    else:
                        f.write(',' + str(scale_feats[key][i]))
            f.write('\n')


if __name__ == '__main__':
    main('Test_Data/id_time_vitals_test.csv',
         'Test_Data/scaled_vitals_test.csv')
