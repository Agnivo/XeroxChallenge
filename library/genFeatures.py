import pandas as pa
import numpy as np
# import math
import datetime
import time


debug = False


def writecsvline(trainfeats, allFeats):
    for feats in allFeats:
        for i in range(len(feats)):
            if i < len(feats) - 1:
                trainfeats.write(str(feats[i]) + ",")
            else:
                trainfeats.write(str(feats[i]) + "\n")
    trainfeats.close()


def generateFeatures(
    vital_file='../Training_Dataset/id_time_vitals_train.csv',
    lab_file='../Training_Dataset/id_time_labs_train.csv',
    age_file='../Training_Dataset/id_age_train.csv',
    label_file='../Training_Dataset/id_label_train.csv',
    method='spline',
    order=3,
    prefix=''
):

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

    ids = np.asarray(np.unique(np.asarray(vitals['ID'])))
    idAges = {}
    for i, row in enumerate(np.asarray(ages)):
        idAges[row[0].astype(np.int32)] = row[1].astype(
            np.int32)
    idLabels = {}
    for i, row in enumerate(np.asarray(labels)):
        idLabels[row[0].astype(np.int32)] = row[1].astype(
            np.int32)
    vitalColumns = []
    for i in xrange(vitals.columns.size):
        if vitals.columns[i] != "TIME" and vitals.columns[i] != "ID":
            vitalColumns.append(str(vitals.columns[i]))
    labColumns = []
    for i in xrange(labs.columns.size):
        if labs.columns[i] != "TIME" and labs.columns[i] != "ID":
            labColumns.append(str(labs.columns[i]))

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
            max(maxtime[id], row[1].astype(np.int32))
        if i >= 100 and debug:
            break

    for i, row in enumerate(np.asarray(labs)):
        tlabs[row[0].astype(np.int32)].append(row[2:])
        if i >= 100 and debug:
            break

    featMeans = [0.42314047,  0.34605021,  0.39641699,  0.24909875,
                 0.95751403, 0.90797301, 0.81011303,
                 0.92037061, 0.23100672, 0.18325522,
                 0.66333616, 0.2702671, 0.41290092,
                 0.61907069, 0.64277418, 0.47439935, 0.4450737,
                 0.12507184, 0.45448799, 0.33454829, 0.30246278,
                 0.48594847, 0.66892206, 0.51567691, 0.67519358,
                 0.18367902, 0.65036754, 0.52432508, 0.47141823,
                 0.56842463, 0.24820434, 0.2081694]

    allFeats = []
    allTargets = []
    idCount = 0
    for id in ids:
        if idCount % 10000 == 0:
            print 'Feature Computed for ', idCount, ' users'
        idCount += 1
        times = []
        for timestamp in ttime[id]:
            times.append(datetime.datetime.fromtimestamp(float(timestamp)))
        dictFeatures = {}
        for i in range(len(vitalColumns)):
            dictFeatures[vitalColumns[i]] = []
        for tFeats in tvitals[id]:
            for i in range(len(tFeats)):
                dictFeatures[vitalColumns[i]].append(tFeats[i])
        for i in range(len(labColumns)):
            dictFeatures[labColumns[i]] = []
        for tFeats in tlabs[id]:
            for i in range(len(tFeats)):
                dictFeatures[labColumns[i]].append(tFeats[i])
        ts = {}
        timeStamps = []
        Columns = []
        for column in vitalColumns:
            Columns.append(column)
        for column in labColumns:
            Columns.append(column)
        for i in range(len(Columns)):
            ts[Columns[i]] = pa.Series(
                dictFeatures[Columns[i]], index=times)
            ts[Columns[i]] = ts[Columns[i]].resample(
                '60T', how='mean')
            if len(timeStamps) == 0:
                for j, row in ts[Columns[0]].iteritems():
                    timeStamps.append(j)
            if debug is True:
                print Columns[i]
                print "Original Series"
                print ts[Columns[i]]
            flag = False
            for j, row in ts[Columns[i]].iteritems():
                if np.isnan(row):
                    flag = True
                break
            if flag is True:
                ts[Columns[i]] = ts[Columns[i]].fillna(
                    value=featMeans[i], limit=1)
            knownVal = featMeans[i]
            for j, row in ts[Columns[i]].iteritems():
                if np.isnan(row) == 0:
                    knownVal = row
            if np.isnan(ts[Columns[i]].loc[
             timeStamps[ts[Columns[i]].size - 1]]):
                ts[Columns[i]].loc[timeStamps[
                 ts[Columns[i]].size - 1]] = knownVal
            if debug is True:
                print "Series interpolated (first nan to mean and last nan)."
                print ts[Columns[i]]
            if (ts[Columns[i]].size - ts[Columns[i]].isnull().sum()) == 3:
                ts[Columns[i]] = ts[Columns[i]].interpolate(
                 method=method, order=2)
            elif (ts[Columns[i]].size - ts[Columns[i]].isnull().sum()) == 2:
                ts[Columns[i]] = ts[Columns[i]].interpolate(
                 method=method, order=1)
            elif (ts[Columns[i]].size - ts[Columns[i]].isnull().sum()) == 1:
                ts[Columns[i]] = ts[Columns[i]].fillna(featMeans[i])
            else:
                ts[Columns[i]] = ts[Columns[i]].interpolate(
                 method=method, order=order)
            if debug is True:
                print "Series after interpolation"
                print ts[Columns[i]]
        j = 0
        for timeStamp in timeStamps:
            feats = []
            sample = {}
            for i in range(len(Columns)):
                feats.append(ts[Columns[i]].loc[timeStamp])
                sample[Columns[i]] = []
            if j == 0:
                for s in range(len(Columns)):
                    sample[Columns[s]].append(featMeans[s])
            elif j < 10:
                for i in range(j):
                    for s in range(len(Columns)):
                        sample[Columns[s]].append(allFeats[i][s])
            else:
                i = j - 10
                while i < j:
                    for s in range(len(Columns)):
                        sample[Columns[s]].append(allFeats[i][s])
                    i += 1
            meanVals = {}
            maxVals = {}
            minVals = {}
            for s in range(len(Columns)):
                meanVals[Columns[s]] = np.nanmean(
                    np.asarray(sample[Columns[s]]))
                maxVals[Columns[s]] = np.nanmax(
                    np.asarray(sample[Columns[s]]))
                minVals[Columns[s]] = np.nanmin(
                    np.asarray(sample[Columns[s]]))
            for s in range(len(Columns)):
                feats.append(meanVals[Columns[s]])
            for s in range(len(Columns)):
                feats.append(maxVals[Columns[s]])
            for s in range(len(Columns)):
                feats.append(minVals[Columns[s]])
            feats.append(idAges[id])
            if len(feats) != 129:
                print "Error in features : Length : ", len(feats),\
                 ", features : ", feats
            targets = []
            targets.append(id.astype(np.int32))
            targets.append(time.mktime(timeStamp.timetuple()).astype(np.int32))
            if idLabels[id] == 1:
                targets.append(1)
            else:
                targets.append(0)
            targets.append(ts['ICU'].loc[timeStamp].astype(np.int32))
            allFeats.append(feats)
            allTargets.append(targets)
            j += 1
        if debug is True:
            for i in range(len(timeStamps)):
                print timeStamps[i], allFeats[i]
            break

    trainfeats = open(prefix + 'train_feats.csv', 'w')
    for i in xrange(129):
        if i < 128:
            trainfeats.write('feat{0},'.format(i))
        else:
            trainfeats.write('feat{0}'.format(i))

    trainfeats.write('\n')
    writecsvline(trainfeats, allFeats)
    traintargets = open(prefix + 'train_targets.csv', 'w')
    traintargets.write('ID,TIME,LABEL,ICU\n')
    writecsvline(traintargets, allTargets)
    

def main():
    generateFeatures()


if __name__ == "__main__":
    main()
