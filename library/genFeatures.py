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
    vital_file='./Training_Dataset/id_time_vitals_train.csv',
    lab_file='./Training_Dataset/id_time_labs_train.csv',
    age_file='./Training_Dataset/id_age_train.csv',
    label_file='./Training_Dataset/id_label_train.csv',
    method='linear',
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
    ttimeset = [set() for i in xrange(np.max(ids) + 1)]
    ticu = [[] for i in xrange(np.max(ids) + 1)]
    maxtime = [0 for i in xrange(np.max(ids) + 1)]

    for i, row in enumerate(np.asarray(vitals)):
        if i % 10000 == 0:
            print 'Pre doing {}'.format(i)
        id = row[0].astype(np.int32)
        tvitals[id].append(row[2:])
        ttime[id].\
            append(row[1].astype(np.int32))
        currTime = datetime.datetime.fromtimestamp(float(row[1]))
        currTime -= datetime.datetime.fromtimestamp(0)
        currTime += datetime.datetime.fromtimestamp(-1800)
        currTime = currTime.replace(minute=0, second=0)
        ttimeset[id].add(currTime)
        ticu[id].\
            append(row[8].astype(np.int32))
        maxtime[id] = \
            max(maxtime[id], row[1].astype(np.int32))
        if i >= 220 and debug:
            break

    for i, row in enumerate(np.asarray(labs)):
        tlabs[row[0].astype(np.int32)].append(row[2:])
        if i >= 220 and debug:
            break

    featMeans = [0.5 for i in xrange(32)]

    allFeats = []
    allTargets = []
    idCount = 0

    then = time.time()

    for id in ids:
        print 'Feature Computed for ', idCount, ' users'
        idCount += 1
        print 'Time remaining {} mins'.format(
            ((time.time() - then) * 1.0
             / idCount * (3600 - idCount)) / 60
        )
        times = []
        for timestamp in ttime[id]:
            currTime = datetime.datetime.fromtimestamp(float(timestamp))
            currTime -= datetime.datetime.fromtimestamp(0)
            currTime += datetime.datetime.fromtimestamp(-1800)
            times.append(currTime)
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
        tempFeats = []
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
                        sample[Columns[s]].append(tempFeats[i][s])
            else:
                i = j - 10
                while i < j:
                    for s in range(len(Columns)):
                        sample[Columns[s]].append(tempFeats[i][s])
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
            targets.append(np.int32(time.mktime(timeStamp.timetuple()) + 1800))
            if idLabels[id] == 1:
                targets.append(1)
            else:
                targets.append(0)
            targets.append(ts['ICU'].loc[timeStamp].astype(np.int32))
            tempFeats.append(feats)
            if True:  # timeStamp in ttimeset[id]:
                allFeats.append(feats)
                allTargets.append(targets)
            else:
                # print timeStamp, "- time not in user with id", id
                pass
            j += 1
        if debug is True:
            j = 0
            for i in range(len(timeStamps)):
                print timeStamps[i]
                if timeStamps[i] in ttimeset[id]:
                    print allFeats[j]
                    j += 1
            if id == 2:
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


def generateTestFeatures(
    vital_file='./Validation_Data/id_time_vitals_val.csv',
    lab_file='./Validation_Data/id_time_labs_val.csv',
    age_file='./Validation_Data/id_age_val.csv',
    prefix='',
    method='linear',
    order=3
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
    # labels = pa.read_csv(
    #     label_file
    # )

    ids = np.asarray(np.unique(np.asarray(vitals['ID'])))
    idAges = {}
    for i, row in enumerate(np.asarray(ages)):
        idAges[row[0].astype(np.int32)] = row[1].astype(
            np.int32)
    # idLabels = {}
    # for i, row in enumerate(np.asarray(labels)):
    #     idLabels[row[0].astype(np.int32)] = row[1].astype(
    #         np.int32)
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
    ttimeset = [set() for i in xrange(np.max(ids) + 1)]
    ticu = [[] for i in xrange(np.max(ids) + 1)]
    maxtime = [0 for i in xrange(np.max(ids) + 1)]

    for i, row in enumerate(np.asarray(vitals)):
        if i % 10000 == 0:
            print 'Pre doing {}'.format(i)
        id = row[0].astype(np.int32)
        tvitals[id].append(row[2:])
        ttime[id].\
            append(row[1].astype(np.int32))
        currTime = datetime.datetime.fromtimestamp(float(row[1]))
        currTime -= datetime.datetime.fromtimestamp(0)
        currTime += datetime.datetime.fromtimestamp(-1800)
        currTime = currTime.replace(minute=0, second=0)
        ttimeset[id].add(currTime)
        ticu[id].\
            append(row[8].astype(np.int32))
        maxtime[id] = \
            max(maxtime[id], row[1].astype(np.int32))
        if i >= 220 and debug:
            break

    for i, row in enumerate(np.asarray(labs)):
        tlabs[row[0].astype(np.int32)].append(row[2:])
        if i >= 220 and debug:
            break

    featMeans = [0.5 for i in xrange(32)]

    allFeats = []
    allTargets = []
    idCount = 0

    then = time.time()
    exampleCount = 0
    for id in ids:
        print 'Feature Computed for ', idCount, ' users and ',\
            exampleCount, 'examples'
        idCount += 1
        print 'Time remaining {} mins'.format(
            ((time.time() - then) * 1.0
             / idCount * (1300 - idCount)) / 60
        )
        times = []
        dictFeatures = {}
        for i in range(len(vitalColumns)):
            dictFeatures[vitalColumns[i]] = []
        for i in range(len(labColumns)):
            dictFeatures[labColumns[i]] = []
        Columns = []
        for column in vitalColumns:
            Columns.append(column)
        for column in labColumns:
            Columns.append(column)
        for k in range(len(tvitals[id])):
            exampleCount += 1
            timestamp = ttime[id][k]
            currTime = datetime.datetime.fromtimestamp(float(timestamp))
            currTime -= datetime.datetime.fromtimestamp(0)
            currTime += datetime.datetime.fromtimestamp(-1800)
            times.append(currTime)
            for j in range(len(vitalColumns)):
                dictFeatures[vitalColumns[j]].append(tvitals[id][k][j])
            for j in range(len(labColumns)):
                dictFeatures[labColumns[j]].append(tlabs[id][k][j])
        ts = {}
        timeStamps = []
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
                # print "Original Series"
                # print ts[Columns[i]]
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
            # if debug is True:
            #     print "Series interpolated \
            #      (first nan to mean and last nan)."
            #     print ts[Columns[i]]
            if (ts[Columns[i]].size - ts[Columns[i]].isnull().sum()) == 3:
                ts[Columns[i]] = ts[Columns[i]].interpolate(
                    method=method, order=2)
            elif (ts[Columns[i]].size - ts[Columns[i]].isnull().sum(
            )) == 2:
                ts[Columns[i]] = ts[Columns[i]].interpolate(
                    method=method, order=1)
            elif (ts[Columns[i]].size - ts[Columns[i]].isnull().sum(
            )) == 1:
                ts[Columns[i]] = ts[Columns[i]].fillna(featMeans[i])
            else:
                ts[Columns[i]] = ts[Columns[i]].interpolate(
                    method=method, order=order)
            # if debug is True:
            #     print "Series after interpolation"
            #     print ts[Columns[i]]
        if debug is True:
            print "TimeStamps : ", timeStamps
        for k in range(len(tvitals[id])):
            sample = {}
            timestamp = ttime[id][k]
            currTime = datetime.datetime.fromtimestamp(float(timestamp))
            currTime -= datetime.datetime.fromtimestamp(0)
            currTime += datetime.datetime.fromtimestamp(-1800)
            currTime = currTime.replace(minute=0, second=0)
            loc = timeStamps.index(currTime)
            if debug is True:
                print "Example {}".format(k), currTime, "hour : ", loc
            for s in range(len(Columns)):
                sample[Columns[s]] = []
            if loc == 0:
                for s in range(len(Columns)):
                    sample[Columns[s]].append(featMeans[s])
            elif loc < 10:
                for i in range(loc):
                    for s in range(len(Columns)):
                        sample[Columns[s]].append(
                            ts[Columns[s]].loc[timeStamps[i]])
                if debug is True:
                    # print "TimeStamps : ", timeStamps
                    print "10 window sample : ", sample

            else:
                i = loc - 10
                while i < loc:
                    for s in range(len(Columns)):
                        sample[Columns[s]].append(
                            ts[Columns[s]].loc[timeStamps[i]])
                    i += 1
                if debug is True:
                    # print "TimeStamps : ", timeStamps
                    print "10 window sample : ", sample

            meanVals = {}
            maxVals = {}
            minVals = {}
            feats = []
            for s in range(len(tvitals[id][k])):
                if np.isnan(tvitals[id][k][s]):
                    feats.append(
                        ts[vitalColumns[s]].loc[timeStamps[loc]])
                else:
                    feats.append(tvitals[id][k][s])
            for s in range(len(tlabs[id][k])):
                if np.isnan(tlabs[id][k][s]):
                    feats.append(
                        ts[labColumns[s]].loc[timeStamps[loc]])
                else:
                    feats.append(tlabs[id][k][s])
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
            targets.append(timestamp)
            targets.append(ticu[id][k].astype(np.int32))
            allFeats.append(feats)
            allTargets.append(targets)
            if debug is True:
                print "Features : ", feats
                print "Targets : ", targets
        if debug is True:
            print "Features : ", len(allFeats), ", Targets : ", len(allTargets)
            # for j in range(len(allFeats)):
            #     print allFeats[j], allTargets[j]
            if id == 3595:
                break

    trainfeats = open(prefix + 'val_feats.csv', 'w')
    for i in xrange(129):
        if i < 128:
            trainfeats.write('feat{0},'.format(i))
        else:
            trainfeats.write('feat{0}'.format(i))

    trainfeats.write('\n')
    writecsvline(trainfeats, allFeats)
    traintargets = open(prefix + 'val_targets.csv', 'w')
    traintargets.write('ID,TIME,ICU\n')
    writecsvline(traintargets, allTargets)


def main():
    # generateFeatures(
    #     vital_file='./Training_Dataset/scaled_vitals_train.csv',
    #     lab_file='./Training_Dataset/scaled_labs_train.csv',
    #     prefix='./Training_Dataset/v2_'
    # )
    generateTestFeatures(
        vital_file='./Test_Data/scaled_vitals_test.csv',
        lab_file='./Test_Data/scaled_labs_test.csv',
        age_file='./Test_Data/id_age_test.csv',
        prefix='./Test_Data/'
    )

if __name__ == "__main__":
    main()
