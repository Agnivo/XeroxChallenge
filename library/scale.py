import pandas as pa
import numpy as np
import sys
import statistics
import math

debug = False
meanByMaxThres = 0.1

def testCode() :
	print "testing"

def writeCSV(fileObj, feats, columns) :
	for column in columns :
		fileObj.write(column + ",")
	fileObj.write('\n')
	for i in range(len(feats['ID'])) :
		for column in columns :
			fileObj.write(feats[column][i] + ',')
		fileObj.write('\n')
	fileObj.close()

def scale() :
	vitals = pa.read_csv(
		'../Training_Dataset/id_time_vitals_train.csv',
		dtype={'ID': np.int32, 'TIME': np.int32, 'ICU': np.int32}
	)
	labs = pa.read_csv(
		'../Training_Dataset/id_time_labs_train.csv',
		dtype={'ID': np.int32, 'TIME': np.int32}
	)
	ages = pa.read_csv(
		'../Training_Dataset/id_age_train.csv',
		dtype={'ID': np.int32, 'AGE': np.int32}
	)
	
	vitalColumns = [str(vitals.columns[i]) for i in xrange(vitals.columns.size)]
	print vitalColumns

	vitalFeats = {}
	for vitalColumn in vitalColumns :
		vitalFeats[vitalColumn] = np.asarray(vitals[vitalColumn]).tolist()
	
	labColumns = [str(labs.columns[i]) for i in xrange(labs.columns.size)]
	print labColumns

	labFeats = {}
	for labColumn in labColumns :
		labFeats[labColumn] = np.asarray(labs[labColumn]).tolist()

	ageColumns = [str(ages.columns[i]) for i in xrange(ages.columns.size)]
	print ageColumns

	ageFeats = {}
	for ageColumn in ageColumns :
		ageFeats[ageColumn] = np.asarray(ages[ageColumn]).tolist()

	for vitalColumn in vitalFeats :
		# print vitalColumn
		if vitalColumn == "ID" or vitalColumn == "ICU" or vitalColumn == "TIME" :
			# print "Entered"
			continue
		finiteValList = []
		for i in range(len(vitalFeats[vitalColumn])) :
			if np.isnan(vitalFeats[vitalColumn][i]) == 0 :
				finiteValList.append(vitalFeats[vitalColumn][i])
		if len(finiteValList) == 0 :
			print vitalColumn, "is all NA"
			continue
		meanValue = statistics.mean(finiteValList)
		maxValue = max(finiteValList)
		print "Column : ", vitalColumn, "max Value : ", maxValue, "mean Value : ", meanValue, "Mean by Max ratio : ", (float(meanValue)/maxValue)
		if (float(meanValue)/maxValue) < meanByMaxThres :
			for i in range(len(vitalFeats[vitalColumn])) :
				if np.isnan(vitalFeats[vitalColumn][i]) == 0 :
					vitalFeats[vitalColumn][i] = math.log(vitalFeats[vitalColumn][i])
			finiteValList = []
			for i in range(len(vitalFeats[vitalColumn])) :
				if np.isnan(vitalFeats[vitalColumn][i]) == 0 :
					finiteValList.append(vitalFeats[vitalColumn][i])
			minValue = min(finiteValList)
			for i in range(len(vitalFeats[vitalColumn])) :
				if np.isnan(vitalFeats[vitalColumn][i]) == 0 :
					vitalFeats[vitalColumn][i] -= minValue
			finiteValList = []
			for i in range(len(vitalFeats[vitalColumn])) :
				if np.isnan(vitalFeats[vitalColumn][i]) == 0 :
					finiteValList.append(vitalFeats[vitalColumn][i])
			meanVal = statistics.mean(finiteValList)
			maxVal = max(finiteValList)
			print "Column : ", vitalColumn, "Mean by Max ratio : ", (float(meanVal)/maxVal)
			for i in range(len(vitalFeats[vitalColumn])) :
				if np.isnan(vitalFeats[vitalColumn][i]) == 0 :
					vitalFeats[vitalColumn][i] /= maxVal
		else :
			print "Column : ", vitalColumn, "Mean by Max ratio : ", (float(meanValue)/maxValue)
			for i in range(len(vitalFeats[vitalColumn])) :
				if np.isnan(vitalFeats[vitalColumn][i]) == 0 :
					vitalFeats[vitalColumn][i] = (float(vitalFeats[vitalColumn][i])/maxValue)

	for labColumn in labFeats :
		if labColumn == "ID" or labColumn == "ICU" or labColumn == "TIME" :
			continue
		finiteValList = []
		for i in range(len(labFeats[labColumn])) :
			if np.isnan(labFeats[labColumn][i]) == 0 :
				finiteValList.append(labFeats[labColumn][i])
		meanValue = statistics.mean(finiteValList)
		maxValue = max(finiteValList)
		if len(finiteValList) == 0 :
			print labColumn, "is all NA"
			continue
		print "Column : ", labColumn, "max Value : ", maxValue, "mean Value : ", meanValue, "Mean by Max ratio : ", (float(meanValue)/maxValue), "finite values : ", len(finiteValList)
		if (float(meanValue)/maxValue) < meanByMaxThres :
			for i in range(len(labFeats[labColumn])) :
				if np.isnan(labFeats[labColumn][i]) == 0 :
					labFeats[labColumn][i] = math.log(labFeats[labColumn][i])
			finiteValList = []
			for i in range(len(labFeats[labColumn])) :
				if np.isnan(labFeats[labColumn][i]) == 0 :
					finiteValList.append(labFeats[labColumn][i])
			minValue = min(finiteValList)
			for i in range(len(labFeats[labColumn])) :
				if np.isnan(labFeats[labColumn][i]) == 0 :
					labFeats[labColumn][i] -= minValue
			finiteValList = []
			for i in range(len(labFeats[labColumn])) :
				if np.isnan(labFeats[labColumn][i]) == 0 :
					finiteValList.append(labFeats[labColumn][i])
			meanVal = statistics.mean(finiteValList)
			maxVal = max(finiteValList)
			print "Column : ", labColumn, "Mean by Max ratio : ", (float(meanVal)/maxVal)
			for i in range(len(labFeats[labColumn])) :
				if np.isnan(labFeats[labColumn][i]) == 0 :
					labFeats[labColumn][i] /= maxVal
		else :
			print "Column : ", labColumn, "Mean by Max ratio : ", (float(meanValue)/maxValue)
			for i in range(len(labFeats[labColumn])) :
				if np.isnan(labFeats[labColumn][i]) == 0 :
					labFeats[labColumn][i] = (float(labFeats[labColumn][i])/maxValue)

	ageFeatMax = max(ageFeats['AGE'])
	for i in range(len(ageFeats['AGE'])) :
		ageFeats['AGE'][i] /= ageFeatMax

	ageFeatsFile = open("Training_Dataset/age_train.csv", 'w')
	writeCSV(ageFeatsFile, ageFeats, ageColumns)

	labFeatsFile = open("Training_Dataset/lab_train.csv", 'w')
	writeCSV(labFeatsFile, labFeats, labColumns)

	vitalFeatsFile = open("Training_Dataset/vital_train.csv", 'w')
	writeCSV(vitalFeatsFile, vitalFeats, vitalColumns)

def main() :
	if debug == True :
		testCode()
	scale()

if __name__ == "__main__" :
	main()