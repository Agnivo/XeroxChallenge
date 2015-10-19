import pandas as pa
import numpy as np
import sys

'''Usage : python interpolate.py $vitals_train_FILE$ $labs_train_file$'''


def writeCSV(fileObj, feats, columns):
    for i in xrange(len(columns)):
        if i < len(columns) - 1:
            fileObj.write(str(columns[i]) + ",")
        else:
            fileObj.write(str(columns[i]))

    fileObj.write('\n')
    for i, row in enumerate(feats.iterrows()):
        for j in range(len(columns)):
            if j != len(columns) - 1:
                fileObj.write(str(row[1][columns[j]]) + ',')
            else:
                fileObj.write(str(row[1][columns[j]]))
        fileObj.write('\n')
    fileObj.close()


def interpolate(vitalfile, labsfile):
    vitals = pa.read_csv(
        vitalfile,
        dtype={'ID': np.int32, 'TIME': np.int32, 'ICU': np.int32}
    )
    labs = pa.read_csv(
        labsfile,
        dtype={'ID': np.int32, 'TIME': np.int32}
    )

    # ids = np.asarray(np.unique(np.asarray(vitals['ID'])))

    prevValues = {}
    vitalColumns = [str(vitals.columns[i])
                    for i in xrange(vitals.columns.size)]

    for vitalColumn in vitalColumns:
        prevValues[vitalColumn] = np.nan

    for i, row in enumerate(vitals.iterrows()):
        for vitalColumn in vitalColumns:
            if vitalColumn == "ID" or vitalColumn == "ICU" or\
               vitalColumn == "TIME":
                continue
            if np.isnan(row[1][vitalColumn]) == 1:
                if np.isnan(prevValues[vitalColumn]) == 0:
                    vitals[vitalColumn][i] = prevValues[vitalColumn]
            else:
                prevValues[vitalColumn] = row[1][vitalColumn]
    outFile = open('../Training_Dataset/interpolated_vitals.csv', 'w')
    writeCSV(outFile, vitals, vitalColumns)
    labColumns = [str(labs.columns[i])
                  for i in xrange(labs.columns.size)]

    prevValues = {}
    for labColumn in labColumns:
        prevValues[labColumn] = np.nan

    for i, row in enumerate(labs.iterrows()):
        for labColumn in labColumns:
            if labColumn == "ID" or labColumn == "ICU" or\
               labColumn == "TIME":
                continue
            if np.isnan(row[1][labColumn]) == 1:
                if np.isnan(prevValues[labColumn]) == 0:
                    labs[labColumn][i] = prevValues[labColumn]
            else:
                prevValues[labColumn] = row[1][labColumn]
    outFile = open('../Training_Dataset/interpolated_labs.csv', 'w')
    writeCSV(outFile, labs, labColumns)


def main():
    vitalfile = sys.argv[1]
    labsfile = sys.argv[2]
    interpolate(vitalfile, labsfile)

if __name__ == "__main__":
    main()
