import pandas as pa
import numpy as np
import random
# import matplotlib.pyplot as py

debug = True


def getfeatures(valfold=0):

    print 'Getting features...'

    vitals = pa.read_csv(
        'Training_Dataset/id_time_vitals_train.csv',
        dtype={'ID': np.int32, 'TIME': np.int32, 'ICU': np.int32}
    )
    labs = pa.read_csv(
        'Training_Dataset/id_time_labs_train.csv',
        dtype={'ID': np.int32, 'TIME': np.int32}
    )
    ages = pa.read_csv(
        'Training_Dataset/id_age_train.csv',
        dtype={'ID': np.int32, 'AGE': np.int32}
    )
    labels = pa.read_csv(
        'Training_Dataset/id_label_train.csv',
        dtype={'ID': np.int32, 'LABEL': np.int32}
    )

    trainfeats = []
    traintargets = []

    valfeats = []
    valtargets = []

    win = 10

    ids = np.asarray(ages['ID'])

    tvitals = [[] for i in xrange(np.max(ids))]
    tlabs = [[] for i in xrange(np.max(ids))]

    for i, row in enumerate(vitals.iterrows()):
        tvitals[row[1]['ID'].astype(np.int32)].append(np.asarray(row[1][2:]))
        if i >= 3 and debug:
            break

    for i, row in enumerate(labs.iterrows()):
        tlabs[row[1]['ID'].astype(np.int32)].append(np.asarray(row[1][2:]))
        if i >= 3 and debug:
            break

    numfolds = 5
    folds = [random.randint(0, numfolds-1)]

    for it, id in enumerate(ids):
        ivitals = tvitals[id]
        ilabs = tlabs[id]

        ivitals = np.asarray(ivitals)
        ilabs = np.asarray(ilabs)

        feat = [[0 for i in xrange(32)] for j in xrange(win)]
        pres = [[0 for i in xrange(32)] for j in xrange(win)]

        target = np.int32(labels[labels['ID'] == id]['LABEL'][0])

        for i in xrange(ivitals.shape[0]):
            feat = feat[1:]
            pres = pres[1:]

            pres.append([0 for j in xrange(32)])
            feat.append([0 for j in xrange(32)])

            for j in xrange(ivitals.shape[1]):
                if not np.isnan(ivitals[i][j]):
                    feat[-1][j] = ivitals[i][j]
                    pres[-1][j] = 1

            for j in xrange(ilabs.shape[1]):
                if not np.isnan(ilabs[i][j]):
                    feat[-1][j+ivitals.shape[1]] = ilabs[i][j]
                    pres[-1][j+ivitals.shape[1]] = 1

            cfeat = np.asarray(feat).flatten()
            cpres = np.asarray(pres).flatten()

            if folds[it] != valfold:
                trainfeats.append(np.hstack((cfeat, cpres)))
                traintargets.append(target)
            else:
                valfeats.append(np.hstack((cfeat, cpres)))
                valtargets.append(target)

        if debug:
            break

    return trainfeats, traintargets, valfeats, valtargets


def main():
    tfeats, ttargets, vfeats, vtargets = getfeatures()
    # print feats, targets

if __name__ == '__main__':
    main()