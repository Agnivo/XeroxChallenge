import pandas as pa
import numpy as np
import math

debug = False
meanByMaxThres = 0.1


def testCode():
    print "testing"


def writeCSV(fileObj, feats, columns):
    for i in xrange(len(columns)):
        if i < len(columns)-1:
            fileObj.write(str(columns[i]) + ",")
        else:
            fileObj.write(str(columns[i]))

    fileObj.write('\n')
    for i in range(len(feats['ID'])):
        for j in range(len(columns)):
            if j != len(columns) - 1:
                fileObj.write(str(feats[columns[j]][i]) + ',')
            else:
                fileObj.write(str(feats[columns[j]][i]))
        fileObj.write('\n')
    fileObj.close()


def scale():
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

    valvitals = pa.read_csv(
        'Validation_Data/id_time_vitals_val.csv',
        dtype={'ID': np.int32, 'TIME': np.int32, 'ICU': np.int32}
    )
    
    vallabs = pa.read_csv(
        'Validation_Data/id_time_labs_val.csv',
        dtype={'ID': np.int32, 'TIME': np.int32}
    )
    valages = pa.read_csv(
        'Validation_Data/id_age_val.csv',
        dtype={'ID': np.int32, 'AGE': np.int32}
    )

    vitalColumns = [str(vitals.columns[i])
                    for i in xrange(vitals.columns.size)]
    print vitalColumns

    vitalFeats = {}
    valvitalFeats = {}
    for vitalColumn in vitalColumns:
        vitalFeats[vitalColumn] = np.asarray(vitals[vitalColumn]).tolist()
        valvitalFeats[vitalColumn] = np.asarray(valvitals[vitalColumn]).tolist()

    labColumns = [str(labs.columns[i]) for i in xrange(labs.columns.size)]
    print labColumns

    labFeats = {}
    vallabFeats = {}
    for labColumn in labColumns:
        labFeats[labColumn] = np.asarray(labs[labColumn]).tolist()
        vallabFeats[labColumn] = np.asarray(vallabs[labColumn]).tolist()

    ageColumns = [str(ages.columns[i]) for i in xrange(ages.columns.size)]
    print ageColumns

    ageFeats = {}
    valageFeats = {}
    for ageColumn in ageColumns:
        ageFeats[ageColumn] = np.asarray(ages[ageColumn]).tolist()
        valageFeats[ageColumn] = np.asarray(valages[ageColumn]).tolist()

    for vitalColumn in vitalFeats:
        # print vitalColumn
        if vitalColumn == "ID" or\
                vitalColumn == "ICU" or\
                vitalColumn == "TIME":
            # print "Entered"
            continue
        finiteValList = []
        for i in range(len(vitalFeats[vitalColumn])):
            if np.isnan(vitalFeats[vitalColumn][i]) == 0:
                finiteValList.append(vitalFeats[vitalColumn][i])
        if len(finiteValList) == 0:
            print vitalColumn, "is all NA"
            continue
        meanValue = np.mean(np.asarray(finiteValList))
        maxValue = max(finiteValList)
        print "Column : ", vitalColumn,\
            "max Value : ", maxValue,\
            "mean Value : ", meanValue,\
            "Mean by Max ratio : ", (float(meanValue) / maxValue)
        if (float(meanValue) / maxValue) < meanByMaxThres:
            for i in range(len(vitalFeats[vitalColumn])):
                if np.isnan(vitalFeats[vitalColumn][i]) == 0:
                    vitalFeats[vitalColumn][i] = math.log(
                        vitalFeats[vitalColumn][i])
            finiteValList = []
            for i in range(len(vitalFeats[vitalColumn])):
                if np.isnan(vitalFeats[vitalColumn][i]) == 0:
                    finiteValList.append(vitalFeats[vitalColumn][i])
            minValue = min(finiteValList)
            for i in range(len(vitalFeats[vitalColumn])):
                if np.isnan(vitalFeats[vitalColumn][i]) == 0:
                    vitalFeats[vitalColumn][i] -= minValue
            finiteValList = []
            for i in range(len(vitalFeats[vitalColumn])):
                if np.isnan(vitalFeats[vitalColumn][i]) == 0:
                    finiteValList.append(vitalFeats[vitalColumn][i])
            meanVal = np.mean(np.asarray(finiteValList))
            maxVal = max(finiteValList)
            print "Column : ", vitalColumn,\
                "Mean by Max ratio : ", (float(meanVal) / maxVal)
            for i in range(len(vitalFeats[vitalColumn])):
                if np.isnan(vitalFeats[vitalColumn][i]) == 0:
                    vitalFeats[vitalColumn][i] /= float(maxVal)
            # Changes
            for i in range(len(valvitalFeats[vitalColumn])):
                if np.isnan(valvitalFeats[vitalColumn][i]) == 0:
                    valvitalFeats[vitalColumn][i] = (float(math.log(
                        valvitalFeats[vitalColumn][i]) - minValue) / maxVal)
        else:
            print "Column : ", vitalColumn,\
                "Mean by Max ratio : ", (float(meanValue) / maxValue)
            for i in range(len(vitalFeats[vitalColumn])):
                if np.isnan(vitalFeats[vitalColumn][i]) == 0:
                    vitalFeats[vitalColumn][i] = (
                        float(vitalFeats[vitalColumn][i]) / maxValue)
            # Changes
            for i in range(len(valvitalFeats[vitalColumn])):
                if np.isnan(valvitalFeats[vitalColumn][i]) == 0:
                    valvitalFeats[vitalColumn][i] = (
                        float(valvitalFeats[vitalColumn][i]) / maxValue)

    for labColumn in labFeats:
        if labColumn == "ID" or labColumn == "ICU" or labColumn == "TIME":
            continue
        finiteValList = []
        for i in range(len(labFeats[labColumn])):
            if np.isnan(labFeats[labColumn][i]) == 0:
                finiteValList.append(labFeats[labColumn][i])
        meanValue = np.mean(np.asarray(finiteValList))
        maxValue = max(finiteValList)
        if len(finiteValList) == 0:
            print labColumn, "is all NA"
            continue
        print "Column : ", labColumn,\
            "max Value : ", maxValue,\
            "mean Value : ", meanValue,\
            "Mean by Max ratio : ", (float(meanValue) / maxValue),\
            "finite values : ", len(finiteValList)
        if (float(meanValue) / maxValue) < meanByMaxThres:
            for i in range(len(labFeats[labColumn])):
                if np.isnan(labFeats[labColumn][i]) == 0:
                    labFeats[labColumn][i] = math.log(labFeats[labColumn][i])
            finiteValList = []
            for i in range(len(labFeats[labColumn])):
                if np.isnan(labFeats[labColumn][i]) == 0:
                    finiteValList.append(labFeats[labColumn][i])
            minValue = min(finiteValList)
            for i in range(len(labFeats[labColumn])):
                if np.isnan(labFeats[labColumn][i]) == 0:
                    labFeats[labColumn][i] -= minValue
            finiteValList = []
            for i in range(len(labFeats[labColumn])):
                if np.isnan(labFeats[labColumn][i]) == 0:
                    finiteValList.append(labFeats[labColumn][i])
            meanVal = np.mean(np.asarray(finiteValList))
            maxVal = max(finiteValList)
            print "Column : ", labColumn,\
                "Mean by Max ratio : ", (float(meanVal) / maxVal)
            for i in range(len(labFeats[labColumn])):
                if np.isnan(labFeats[labColumn][i]) == 0:
                    labFeats[labColumn][i] /= float(maxVal)
            # Changes
            for i in range(len(vallabFeats[labColumn])):
                if np.isnan(vallabFeats[labColumn][i]) == 0:
                    vallabFeats[labColumn][i] = (float(math.log(
                        vallabFeats[labColumn][i]) - minValue) / maxVal)
        else:
            print "Column : ", labColumn,\
                "Mean by Max ratio : ", (float(meanValue) / maxValue)
            for i in range(len(labFeats[labColumn])):
                if np.isnan(labFeats[labColumn][i]) == 0:
                    labFeats[labColumn][i] = (
                        float(labFeats[labColumn][i]) / maxValue)
            # Changes
            for i in range(len(vallabFeats[labColumn])):
                if np.isnan(vallabFeats[labColumn][i]) == 0:
                    vallabFeats[labColumn][i] = (
                        float(vallabFeats[labColumn][i]) / maxValue)

    finiteValList = []
    for i in range(len(ageFeats['AGE'])):
        if np.isnan(ageFeats['AGE'][i]) == 0:
            finiteValList.append(ageFeats['AGE'][i])
    ageFeatMax = max(finiteValList)
    for i in range(len(ageFeats['AGE'])):
        if np.isnan(ageFeats['AGE'][i]) == 0:
            ageFeats['AGE'][i] /= float(ageFeatMax)

    # Changes
    for i in range(len(valageFeats['AGE'])):
        if np.isnan(valageFeats['AGE'][i]) == 0:
            valageFeats['AGE'][i] /= float(ageFeatMax)

    ageFeatsFile = open("Training_Dataset/age_train.csv", 'w')
    writeCSV(ageFeatsFile, ageFeats, ageColumns)

    labFeatsFile = open("Training_Dataset/lab_train.csv", 'w')
    writeCSV(labFeatsFile, labFeats, labColumns)

    vitalFeatsFile = open("Training_Dataset/vital_train.csv", 'w')
    writeCSV(vitalFeatsFile, vitalFeats, vitalColumns)
    
    # Changes
    valageFeatsFile = open("Validation_Data/age_train.csv", 'w')
    writeCSV(valageFeatsFile, valageFeats, ageColumns)

    vallabFeatsFile = open("Validation_Data/lab_train.csv", 'w')
    writeCSV(vallabFeatsFile, vallabFeats, labColumns)

    valvitalFeatsFile = open("Validation_Data/vital_train.csv", 'w')
    writeCSV(valvitalFeatsFile, valvitalFeats, vitalColumns)

def main():
    if debug is True:
        testCode()
    scale()

if __name__ == "__main__":
    main()