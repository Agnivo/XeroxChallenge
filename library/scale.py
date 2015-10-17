import pandas as pa
import numpy as np
import sys
import statistics

debug = False

def testCode() :
	print "testing"

def scale() :
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


def main() :
	if debug == True :
		testCode()
	scale()

if __name__ == "__main__" :
	main()