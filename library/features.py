import pandas as pa
import numpy as np
# import matplotlib.pyplot as py

debug = False


def writecsvline(fileobj, array):
    fileobj.write(','.join([str(i) for i in array]) + '\n')


def getfeatures(
    vital_file='Training_Dataset/id_time_vitals_train.csv',
    lab_file='Training_Dataset/id_time_labs_train.csv',
    age_file='Training_Dataset/id_age_train.csv',
    label_file='Training_Dataset/id_label_train.csv',
    prefix='',
    valfold=0
):
    '''
    Extracts window features from the given time series and
    saves them to the respective csv files.

    Training features and validation features contain 641 columns each
    denoting the 641 features.
    Training targets and validation targets contain 3 columns each for
    'ID', 'TIME', 'LABEL' and 'ICU'

    Further you might want to normalize the features so that their
    values lie between 0, 1.

    '''

    print 'Getting features...'

    vitals = pa.read_csv(
        vital_file
    )
    labs = pa.read_csv(
        lab_file
    )
    ages = pa.read_csv(
        age_file
    )
    labels = pa.read_csv(
        label_file
    )

    win = 10

    ids = np.asarray(ages['ID'])

    numfolds = 5
    folds = np.random.randint(0, numfolds - 1, np.max(ids) + 1)

    tvitals = [[] for i in xrange(np.max(ids) + 1)]
    tlabs = [[] for i in xrange(np.max(ids) + 1)]
    ttime = [[] for i in xrange(np.max(ids) + 1)]
    ticu = [[] for i in xrange(np.max(ids) + 1)]
    maxtime = [0 for i in xrange(np.max(ids) + 1)]

    for i, row in enumerate(vitals.iterrows()):
        id = row[1]['ID'].astype(np.int32)
        tvitals[id].append(np.asarray(row[1][2:]))
        ttime[id].\
            append(row[1]['TIME'].astype(np.int32))
        ticu[id].\
            append(row[1]['ICU'].astype(np.int32))
        maxtime[id] = \
            max(maxtime[id], row[1]['TIME'].astype(np.int32))
        if i >= 100 and debug:
            break

    for i, row in enumerate(labs.iterrows()):
        tlabs[row[1]['ID'].astype(np.int32)].append(np.asarray(row[1][2:]))
        if i >= 100 and debug:
            break

    trainfeats = open(prefix+'train_feats.csv', 'w')
    traintargets = open(prefix+'train_targets.csv', 'w')

    valfeats = open(prefix+'validation_feats.csv', 'w')
    valtargets = open(prefix+'validation_targets.csv', 'w')

    traintargets.write('ID,TIME,LABEL,ICU\n')
    valtargets.write('ID,TIME,LABEL,ICU\n')

    for i in xrange(641):
        if i < 640:
            trainfeats.write('feat{0},'.format(i))
            valfeats.write('feat{0},'.format(i))
        else:
            trainfeats.write('feat{0}'.format(i))
            valfeats.write('feat{0}'.format(i))

    trainfeats.write('\n')
    valfeats.write('\n')

    for it, id in enumerate(ids):

        print 'Doing', id
        ivitals = tvitals[id]
        ilabs = tlabs[id]

        ivitals = np.asarray(ivitals)
        ilabs = np.asarray(ilabs)

        feat = [[0 for i in xrange(32)] for j in xrange(win)]
        pres = [[0 for i in xrange(32)] for j in xrange(win)]

        target = np.int32(labels[labels['ID'] == id]['LABEL'])[0]
        age = np.int32(ages[ages['ID'] == id]['AGE'])[0]

        for i in xrange(ivitals.shape[0]):
            if target == 0 and i % 3 != 0:
                continue

            time = ttime[id][i]
            icu = ticu[id][i]

            feat = feat[1:]
            pres = pres[1:]

            ttarget = 0
            if target == 1 and maxtime[id] - time <= 36000:
                ttarget = 1

            # pres.append([0 for j in xrange(32)])
            # feat.append([0 for j in xrange(32)])
            feat.append(feat[-1])
            pres.append(pres[-1])

            for j in xrange(ivitals.shape[1]):
                if not np.isnan(ivitals[i][j]):
                    feat[-1][j] = ivitals[i][j]
                    pres[-1][j] = 1

            for j in xrange(ilabs.shape[1]):
                if not np.isnan(ilabs[i][j]):
                    feat[-1][j + ivitals.shape[1]] = ilabs[i][j]
                    pres[-1][j + ivitals.shape[1]] = 1

            cfeat = np.asarray(feat).flatten()
            cpres = np.asarray(pres).flatten()

            if folds[it] != valfold:
                writecsvline(trainfeats, np.hstack((cfeat, cpres, [age])))
                writecsvline(traintargets, [id, time, ttarget, icu])
            else:
                writecsvline(valfeats, np.hstack((cfeat, cpres, [age])))
                writecsvline(valtargets, [id, time, ttarget, icu])

        if debug and id > 2:
            break

    trainfeats.close()
    traintargets.close()
    valfeats.close()
    valtargets.close()


def main():
    getfeatures(
        vital_file='Training_Dataset/vital_train.csv',
        lab_file='Training_Dataset/lab_train.csv',
        age_file='Training_Dataset/age_train.csv',
        label_file='Training_Dataset/id_label_train.csv'
    )

    # print tfeats.shape, ttargets.shape, vfeats.shape, vtargets.shape
    # print feats, targets

if __name__ == '__main__':
    main()
