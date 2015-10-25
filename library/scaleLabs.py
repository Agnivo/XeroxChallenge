import sys
import pandas as pd
import math


def main(filein, fileout):
    train_labs = pd.read_csv(filein)
    min_labs = {}
    max_labs = {}

    min_labs['L1'] = 7.2
    max_labs['L1'] = 7.6

    min_labs['L2'] = 75.0
    max_labs['L2'] = 100.0

    min_labs['L3'] = 35.0
    max_labs['L3'] = 45.0

    min_labs['L4'] = 130.0
    max_labs['L4'] = 150.0

    min_labs['L5'] = 3.5
    max_labs['L5'] = 5.5

    min_labs['L6'] = 20.0
    max_labs['L6'] = 30.0

    min_labs['L7'] = 5.0
    max_labs['L7'] = 20.0

    min_labs['L8'] = 0.5
    max_labs['L8'] = 1.5

    min_labs['L9'] = 4.0
    max_labs['L9'] = 12.0

    min_labs['L10'] = 35.0
    max_labs['L10'] = 55.0

    min_labs['L11'] = 150.0
    max_labs['L11'] = 350.0

    min_labs['L12'] = 0.0
    max_labs['L12'] = 2.0

    min_labs['L13'] = 800.0
    max_labs['L13'] = 2000.0

    min_labs['L14'] = 80.0
    max_labs['L14'] = 120.0

    min_labs['L15'] = 0.5
    max_labs['L15'] = 2.5

    min_labs['L16'] = 0.0
    max_labs['L16'] = 0.1

    min_labs['L17'] = 0.0
    max_labs['L17'] = 0.2

    min_labs['L18'] = 70.0
    max_labs['L18'] = 130.0

    min_labs['L19'] = 70.0
    max_labs['L19'] = 100.0

    min_labs['L20'] = 19.0
    max_labs['L20'] = 22.0

    min_labs['L21'] = 3.0
    max_labs['L21'] = 5.0

    min_labs['L22'] = 32.0
    max_labs['L22'] = 110.0

    min_labs['L23'] = 5.0
    max_labs['L23'] = 40.0

    min_labs['L24'] = 30.0
    max_labs['L24'] = 80.0

    min_labs['L25'] = 1.5
    max_labs['L25'] = 2.5

    key_labs = train_labs.keys()
    num = len(train_labs[key_labs[0]])

    maxval = {}
    nacount = 0

    scale_feats = {}
    for key in key_labs:
        scale_feats[key] = []

    for i in xrange(0, num):
        scale_feats[key_labs[0]].append(train_labs[key_labs[0]][i])
        scale_feats[key_labs[1]].append(train_labs[key_labs[1]][i])
        for key in key_labs[2:]:
            if not math.isnan(train_labs[key][i]):
                if key not in maxval:
                    maxval[key] = train_labs[key][i]
                else:
                    maxval[key] = max(maxval[key], train_labs[key][i])

                scale_feats[key].append(
                    (train_labs[key][i] - min_labs[key]) /
                    (max_labs[key] - min_labs[key])
                )
                if scale_feats[key][i] > 10.0:
                    nacount += 1
            else:
                scale_feats[key].append(train_labs[key][i])

    print maxval
    print nacount

    with open(fileout, 'w') as f:
        f.write(key_labs[0])
        for key in key_labs[1:]:
            f.write(',' + key)
        f.write('\n')

        for i in xrange(0, num):
            f.write(
                str(scale_feats[key_labs[0]][i]) + ',' +
                str(scale_feats[key_labs[1]][i])
            )
            for key in key_labs[2:]:
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
    main('Validation_Data/id_time_labs_val.csv',
         'Validation_Data/scaled_labs_val.csv')
