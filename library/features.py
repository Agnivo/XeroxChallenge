import pandas as pa
import numpy as np
# import matplotlib.pyplot as py

debug = False


def writecsvline(fileobj, array):
    fileobj.write(','.join([str(i) for i in array]) + '\n')


def getfeatures1(
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

    feat_means = [0.42314047,  0.34605021,  0.39641699,  0.24909875,
                  0.95751403, 0.90797301, 0.81011303,
                  0.92037061, 0.23100672, 0.18325522,
                  0.66333616, 0.2702671, 0.41290092,
                  0.61907069, 0.64277418, 0.47439935, 0.4450737,
                  0.12507184, 0.45448799, 0.33454829, 0.30246278,
                  0.48594847, 0.66892206, 0.51567691, 0.67519358,
                  0.18367902, 0.65036754, 0.52432508, 0.47141823,
                  0.56842463, 0.24820434, 0.2081694]

    for i, row in enumerate(np.asarray(vitals)):
        if i % 10000 == 0:
            print 'Pre doing {}'.format(i)
        id = row[0].astype(np.int32)
        tvitals[id].append(row[2:])
        ttime[id].\
            append(row[1].astype(np.int32))
        ticu[id].\
            append(row[8].astype(np.int32))
        maxtime[id] = \
            max(maxtime[id], row[8].astype(np.int32))
        if i >= 100 and debug:
            break

    for i, row in enumerate(np.asarray(labs)):
        tlabs[row[0].astype(np.int32)].append(row[2:])
        if i >= 100 and debug:
            break

    trainfeats = open(prefix + 'train_feats.csv', 'w')
    traintargets = open(prefix + 'train_targets.csv', 'w')

    valfeats = open(prefix + 'validation_feats.csv', 'w')
    valtargets = open(prefix + 'validation_targets.csv', 'w')

    traintargets.write('ID,TIME,LABEL,ICU\n')
    valtargets.write('ID,TIME,LABEL,ICU\n')

    for i in xrange(321):
        if i < 320:
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

        feat = [feat_means for j in xrange(win)]
        # pres = [[0 for i in xrange(32)] for j in xrange(win)]

        target = np.int32(labels[labels['ID'] == id]['LABEL'])[0]
        age = np.int32(ages[ages['ID'] == id]['AGE'])[0]

        for i in xrange(ivitals.shape[0]):

            time = ttime[id][i]
            icu = ticu[id][i]

            feat = feat[1:]
            # pres = pres[1:]

            ttarget = 0
            if target == 1 and maxtime[id] - time <= 10 * 60 * 60:
                ttarget = 1

            # pres.append([0 for j in xrange(32)])
            # feat.append([0 for j in xrange(32)])
            feat.append(feat[-1])
            # pres.append(pres[-1])

            for j in xrange(ivitals.shape[1]):
                if not np.isnan(ivitals[i][j]):
                    feat[-1][j] = ivitals[i][j]
                    # pres[-1][j] = 1

            for j in xrange(ilabs.shape[1]):
                if not np.isnan(ilabs[i][j]):
                    feat[-1][j + ivitals.shape[1]] = ilabs[i][j]
                    # pres[-1][j + ivitals.shape[1]] = 1

            cfeat = np.asarray(feat).flatten()
            # cpres = np.asarray(pres).flatten()

            # if target == 0 and i % 2 != 0:
            #     continue

            if folds[it] != valfold:
                if target == 1 and ttarget == 0:
                    continue
                writecsvline(trainfeats, np.hstack((cfeat, [age])))
                writecsvline(traintargets, [id, time, ttarget, icu])
            else:
                writecsvline(valfeats, np.hstack((cfeat, [age])))
                writecsvline(valtargets, [id, time, ttarget, icu])

        if debug and id > 2:
            break

    trainfeats.close()
    traintargets.close()
    valfeats.close()
    valtargets.close()


def getfeatures12(
    vital_file='Training_Dataset/id_time_vitals_train.csv',
    lab_file='Training_Dataset/id_time_labs_train.csv',
    age_file='Training_Dataset/id_age_train.csv',
    prefix='',
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

    win = 10

    ids = np.asarray(ages['ID'])

    tvitals = [[] for i in xrange(np.max(ids) + 1)]
    tlabs = [[] for i in xrange(np.max(ids) + 1)]
    ttime = [[] for i in xrange(np.max(ids) + 1)]
    ticu = [[] for i in xrange(np.max(ids) + 1)]
    maxtime = [0 for i in xrange(np.max(ids) + 1)]

    feat_means = [0.42314047,  0.34605021,  0.39641699,  0.24909875,
                  0.95751403, 0.90797301, 0.81011303,
                  0.92037061, 0.23100672, 0.18325522,
                  0.66333616, 0.2702671, 0.41290092,
                  0.61907069, 0.64277418, 0.47439935, 0.4450737,
                  0.12507184, 0.45448799, 0.33454829, 0.30246278,
                  0.48594847, 0.66892206, 0.51567691, 0.67519358,
                  0.18367902, 0.65036754, 0.52432508, 0.47141823,
                  0.56842463, 0.24820434, 0.2081694]

    for i, row in enumerate(np.asarray(vitals)):
        if i % 10000 == 0:
            print 'Pre doing {}'.format(i)
        id = row[0].astype(np.int32)
        tvitals[id].append(row[2:])
        ttime[id].\
            append(row[1].astype(np.int32))
        ticu[id].\
            append(row[8].astype(np.int32))
        maxtime[id] = \
            max(maxtime[id], row[8].astype(np.int32))
        if i >= 100 and debug:
            break

    for i, row in enumerate(np.asarray(labs)):
        tlabs[row[0].astype(np.int32)].append(row[2:])
        if i >= 100 and debug:
            break

    trainfeats = open(prefix + 'train_feats.csv', 'w')
    traintargets = open(prefix + 'train_targets.csv', 'w')
    traintargets.write('ID,TIME,ICU\n')

    for i in xrange(321):
        if i < 320:
            trainfeats.write('feat{0},'.format(i))
        else:
            trainfeats.write('feat{0}'.format(i))

    trainfeats.write('\n')

    for it, id in enumerate(ids):

        print 'Doing', id
        ivitals = tvitals[id]
        ilabs = tlabs[id]

        ivitals = np.asarray(ivitals)
        ilabs = np.asarray(ilabs)

        feat = [feat_means for j in xrange(win)]
        # pres = [[0 for i in xrange(32)] for j in xrange(win)]

        age = np.int32(ages[ages['ID'] == id]['AGE'])[0]

        for i in xrange(ivitals.shape[0]):

            time = ttime[id][i]
            icu = ticu[id][i]

            feat = feat[1:]
            # pres = pres[1:]

            # pres.append([0 for j in xrange(32)])
            # feat.append([0 for j in xrange(32)])
            feat.append(feat[-1])
            # pres.append(pres[-1])

            for j in xrange(ivitals.shape[1]):
                if not np.isnan(ivitals[i][j]):
                    feat[-1][j] = ivitals[i][j]
                    # pres[-1][j] = 1

            for j in xrange(ilabs.shape[1]):
                if not np.isnan(ilabs[i][j]):
                    feat[-1][j + ivitals.shape[1]] = ilabs[i][j]
                    # pres[-1][j + ivitals.shape[1]] = 1

            cfeat = np.asarray(feat).flatten()
            # cpres = np.asarray(pres).flatten()

            # if target == 0 and i % 2 != 0:
            #     continue

            writecsvline(trainfeats, np.hstack((cfeat, [age])))
            writecsvline(traintargets, [id, time, icu])

        if debug and id > 2:
            break

    trainfeats.close()
    traintargets.close()


def getfeatures2(
    vital_file='Training_Dataset/id_time_vitals_train.csv',
    lab_file='Training_Dataset/id_time_labs_train.csv',
    age_file='Training_Dataset/id_age_train.csv',
    label_file='Training_Dataset/id_label_train.csv',
    prefix='',
    full=False,
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

    for i, row in enumerate(np.asarray(vitals)):
        if i % 10000 == 0:
            print 'Pre doing {}'.format(i)
        id = row[0].astype(np.int32)
        tvitals[id].append(row[2:])
        ttime[id].\
            append(row[1].astype(np.int32))
        ticu[id].\
            append(row[8].astype(np.int32))
        maxtime[id] = \
            max(maxtime[id], row[8].astype(np.int32))
        if i >= 100 and debug:
            break

    for i, row in enumerate(np.asarray(labs)):
        tlabs[row[0].astype(np.int32)].append(row[2:])
        if i >= 100 and debug:
            break

    feat_means = [0.42314047,  0.34605021,  0.39641699,  0.24909875,
                  0.95751403, 0.90797301, 0.81011303,
                  0.92037061, 0.23100672, 0.18325522,
                  0.66333616, 0.2702671, 0.41290092,
                  0.61907069, 0.64277418, 0.47439935, 0.4450737,
                  0.12507184, 0.45448799, 0.33454829, 0.30246278,
                  0.48594847, 0.66892206, 0.51567691, 0.67519358,
                  0.18367902, 0.65036754, 0.52432508, 0.47141823,
                  0.56842463, 0.24820434, 0.2081694]

    trainfeats = open(prefix + 'train_feats.csv', 'w')
    traintargets = open(prefix + 'train_targets.csv', 'w')
    traintargets.write('ID,TIME,LABEL,ICU\n')

    if not full:
        valfeats = open(prefix + 'validation_feats.csv', 'w')
        valtargets = open(prefix + 'validation_targets.csv', 'w')
        valtargets.write('ID,TIME,LABEL,ICU\n')

    numfeats = 32 * 4 + 1
    for i in xrange(numfeats):
        if i < numfeats - 1:
            trainfeats.write('feat{0},'.format(i))
            if not full:
                valfeats.write('feat{0},'.format(i))
        else:
            trainfeats.write('feat{0}'.format(i))
            if not full:
                valfeats.write('feat{0}'.format(i))

    trainfeats.write('\n')
    if not full:
        valfeats.write('\n')

    for it, id in enumerate(ids):

        print 'Doing', id
        ivitals = tvitals[id]
        ilabs = tlabs[id]

        ivitals = np.asarray(ivitals)
        ilabs = np.asarray(ilabs)

        feat = [[np.nan for i in xrange(32)] for j in xrange(win)]
        # pres = [[0 for i in xrange(32)] for j in xrange(win)]

        target = np.int32(labels[labels['ID'] == id]['LABEL'])[0]
        age = np.int32(ages[ages['ID'] == id]['AGE'])[0]

        for i in xrange(ivitals.shape[0]):

            time = ttime[id][i]
            icu = ticu[id][i]

            feat = feat[1:]
            # pres = pres[1:]

            ttarget = 0
            if target == 1:  # and maxtime[id] - time <= 10 * 60 * 60:
                ttarget = 1

            feat.append(np.hstack((ivitals[i].ravel(), ilabs[i].ravel())))
            for i in xrange(32):
                if np.isnan(feat[-1][i]):
                    feat[-1][i] = feat[-2][i]

            cfeat = np.asarray(feat)

            temp = []
            for j in xrange(32):
                t1 = [cfeat[9, j], np.nanmean(cfeat[:, j]),
                      np.nanmin(cfeat[:, j]), np.nanmax(cfeat[:, j])]
                if np.any(np.isnan(t1)):
                    t1 = [
                        feat_means[j], feat_means[j],
                        feat_means[j], feat_means[j]
                    ]
                temp.append(t1)

            cfeat = np.hstack(temp)

            # cfeat = np.asarray(feat).flatten()
            # cpres = np.asarray(pres).flatten()

            # if target == 0 and i % 2 != 0:
            #     continue

            if folds[it] != valfold or full:
                if target == 1 and ttarget == 0:
                    continue
                writecsvline(trainfeats, np.hstack((cfeat, [age])))
                writecsvline(traintargets, [id, time, ttarget, icu])
            else:
                writecsvline(valfeats, np.hstack((cfeat, [age])))
                writecsvline(valtargets, [id, time, ttarget, icu])

        if debug and id > 2:
            break

    trainfeats.close()
    traintargets.close()
    if not full:
        valfeats.close()
        valtargets.close()


def getfeatures3(
    vital_file='Validation_Data/id_time_vitals_train.csv',
    lab_file='Validation_Data/id_time_labs_train.csv',
    age_file='Validation_Data/id_age_train.csv',
    label_file='Validation_Data/id_label_train.csv',
    prefix='val_'
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

    win = 10

    ids = np.asarray(ages['ID'])

    tvitals = [[] for i in xrange(np.max(ids) + 1)]
    tlabs = [[] for i in xrange(np.max(ids) + 1)]
    ttime = [[] for i in xrange(np.max(ids) + 1)]
    ticu = [[] for i in xrange(np.max(ids) + 1)]
    maxtime = [0 for i in xrange(np.max(ids) + 1)]

    for i, row in enumerate(np.asarray(vitals)):
        if i % 10000 == 0:
            print 'Pre doing {}'.format(i)
        id = row[0].astype(np.int32)
        tvitals[id].append(row[2:])
        ttime[id].\
            append(row[1].astype(np.int32))
        ticu[id].\
            append(row[8].astype(np.int32))
        maxtime[id] = \
            max(maxtime[id], row[8].astype(np.int32))
        if i >= 100 and debug:
            break

    for i, row in enumerate(np.asarray(labs)):
        tlabs[row[0].astype(np.int32)].append(row[2:])
        if i >= 100 and debug:
            break

    feat_means = [0.42314047,  0.34605021,  0.39641699,  0.24909875,
                  0.95751403, 0.90797301, 0.81011303,
                  0.92037061, 0.23100672, 0.18325522,
                  0.66333616, 0.2702671, 0.41290092,
                  0.61907069, 0.64277418, 0.47439935, 0.4450737,
                  0.12507184, 0.45448799, 0.33454829, 0.30246278,
                  0.48594847, 0.66892206, 0.51567691, 0.67519358,
                  0.18367902, 0.65036754, 0.52432508, 0.47141823,
                  0.56842463, 0.24820434, 0.2081694]

    trainfeats = open(prefix + 'train_feats.csv', 'w')
    traintargets = open(prefix + 'train_targets.csv', 'w')
    traintargets.write('ID,TIME,ICU\n')

    numfeats = 32 * 4 + 1
    for i in xrange(numfeats):
        if i < numfeats - 1:
            trainfeats.write('feat{0},'.format(i))
        else:
            trainfeats.write('feat{0}'.format(i))

    trainfeats.write('\n')

    for it, id in enumerate(ids):

        print 'Doing', id
        ivitals = tvitals[id]
        ilabs = tlabs[id]

        ivitals = np.asarray(ivitals)
        ilabs = np.asarray(ilabs)

        feat = [[np.nan for i in xrange(32)] for j in xrange(win)]
        # pres = [[0 for i in xrange(32)] for j in xrange(win)]

        age = np.int32(ages[ages['ID'] == id]['AGE'])[0]

        for i in xrange(ivitals.shape[0]):

            time = ttime[id][i]
            icu = ticu[id][i]

            feat = feat[1:]
            # pres = pres[1:]

            feat.append(np.hstack((ivitals[i].ravel(), ilabs[i].ravel())))
            for i in xrange(32):
                if np.isnan(feat[-1][i]):
                    feat[-1][i] = feat[-2][i]

            cfeat = np.asarray(feat)

            temp = []
            for j in xrange(32):
                t1 = [cfeat[9, j], np.nanmean(cfeat[:, j]),
                      np.nanmin(cfeat[:, j]), np.nanmax(cfeat[:, j])]
                if np.any(np.isnan(t1)):
                    t1 = [
                        feat_means[j], feat_means[j],
                        feat_means[j], feat_means[j]
                    ]
                temp.append(t1)

            cfeat = np.hstack(temp)

            # cfeat = np.asarray(feat).flatten()
            # cpres = np.asarray(pres).flatten()

            # if target == 0 and i % 2 != 0:
            #     continue

            writecsvline(trainfeats, np.hstack((cfeat, [age])))
            writecsvline(traintargets, [id, time, icu])

        if debug and id > 2:
            break

    trainfeats.close()
    traintargets.close()


def main():
    # getfeatures1(
    #     vital_file='Training_Dataset/vital_train.csv',
    #     lab_file='Training_Dataset/lab_train.csv',
    #     age_file='Training_Dataset/age_train.csv',
    #     label_file='Training_Dataset/id_label_train.csv',
    #     prefix='win_',
    #     valfold=1
    # )

    # getfeatures12(
    #     vital_file='Validation_Data/vital_train.csv',
    #     lab_file='Validation_Data/lab_train.csv',
    #     age_file='Validation_Data/age_train.csv',
    #     prefix='win_val_'
    # )

    # getfeatures2(
    #     vital_file='Training_Dataset/vital_train.csv',
    #     lab_file='Training_Dataset/lab_train.csv',
    #     age_file='Training_Dataset/age_train.csv',
    #     label_file='Training_Dataset/id_label_train.csv'
    # )

    getfeatures3(
        vital_file='Test_Data/vital_train.csv',
        lab_file='Test_Data/lab_train.csv',
        age_file='Test_Data/age_train.csv',
        prefix='test_'
    )

if __name__ == '__main__':
    main()
