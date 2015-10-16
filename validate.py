import pandas as pa
import numpy as np
import sys
import statistics

''' Usage : python validate.py Predicted.csv Training_Dataset/actualLabel.csv
	Format of Predicted.csv : ID,TIME,ICU,LABEL
	Actual Labels read from Training_Dataset/actualLabel.csv, Format : ID,LABEL
'''

debug = False

''' Assuming 1. All the prediction labels are either 0 or 1 <=> 
	All labels over all predictions are either 0 or 1
'''
def checkAllLabels(predictedData) :
	j = 0
	firstLabel = 0
	flag = 1
	for i, row in enumerate(predictedData.iterrows()) :
		if j == 0 :
			firstLabel = row[1]['LABEL']
		else :
			if firstLabel != row[1]['LABEL'] :
				flag = 0
				break
		j += 1
	return flag

''' Assuming 2.Predictions for all patients in validation/test are provided <=>
	All patients have finalPrediction as 1 
'''
def checkAllPredicted(finalPrediction) :
	for i in finalPrediction :
		if (finalPrediction[i] == 0) :
			return 0
	return 1

''' Assuming 3. For each patient, predictions are present for all timestamps where ICU flag == 1 <=>
	All patients have prediction = 1 when ICUFlag == 1, i.e. if any patient 
	has prediction = 0 when ICUFlag == 1, return false
'''
def checkAllICUFlags(predictedData) :
	for i, row in enumerate(predictedData.iterrows()) :
		if (row[1]['ICU'] == 1 and row[1]['LABEL'] == 0) :
			return 0
	return 1

''' Assuming 4. For each patient, predictions are present for only those timestamps where ICU flag == 1 <=>
	Some patients have prediction = 1 when ICUFlag == 0, return false, else, returm true
'''
def checkOnlyICUFlags(predictedData) :
	for i, row in enumerate(predictedData.iterrows()) :
		if (row[1]['ICU'] == 0 and row[1]['LABEL'] == 1) :
			return 0
	return 1

def validate() :
	predictedData = pa.read_csv(
				sys.argv[1], 
				dtype={'ID': np.int32, 'TIME': np.int32, 'ICU': np.int32, 'LABEL': np.int32}
			)

	actualData = pa.read_csv(
        sys.argv[2],
        dtype={'ID': np.int32, 'LABEL': np.int32}
    )

	ids = np.asarray(predictedData['ID'])
	minTime = np.max(np.asarray(predictedData['TIME'])) + 1
	maxTime = 0
	finalPrediction = {}
	minTimes = {}
	maxTimes = {}
	finalPredictedTrue = []

	for (x, i) in np.ndenumerate(ids) :
		finalPrediction[i] = 0
		minTimes[i] = minTime
		maxTimes[i] = maxTime

	for i, row in enumerate(predictedData.iterrows()) :
		if (row[1]['ICU'] == 1 and row[1]['LABEL'] == 1) :
			finalPrediction[row[1]['ID']] = 1
			if (row[1]['TIME'] > maxTimes[row[1]['ID']]) :
				maxTimes[row[1]['ID']] = row[1]['TIME']
			if (row[1]['TIME'] < minTimes[row[1]['ID']]) :
				minTimes[row[1]['ID']] = row[1]['TIME']
			if row[1]['ID'] not in finalPredictedTrue :
				finalPredictedTrue.append(row[1]['ID'])

	predictionTime = {}
	for (x, i) in np.ndenumerate(ids) :
		if i not in finalPredictedTrue :
			predictionTime[i] = -1
		else :
			predictionTime[i] = maxTimes[i] - minTimes[i]

	actualLabels = {}
	for i, row in enumerate(actualData.iterrows()) :
		actualLabels[row[1]['ID']] = row[1]['LABEL']

	tp = tn = fp = fn = 0
	for (x, i) in np.ndenumerate(ids) :
		if actualLabels[i] == 1 and finalPrediction[i] == 1 :
			tp += 1
		if actualLabels[i] == 0 and finalPrediction[i] == 0 :
			tn += 1					
		if actualLabels[i] == 0 and finalPrediction[i] == 1 :
			fp += 1
		if actualLabels[i] == 1 and finalPrediction[i] == 0 :
			fn += 1

	print "tp : ", tp, ", tn : ", tn, ", fp : ", fp, ", fn : ", fn
	if tp + fn != 0 :
		sensitivity = (float(tp)/(tp + fn))
	else :
		sensitivity = 0
	if tn + fp != 0 :
		specificity = (float(tn)/(tn + fp))
	else :
		specificity = 0
	accuracy = (float(tp + tn)/(tp + tn + fp + fn))

	predictionTimesList = []
	for i in predictionTime :
		if predictionTime[i] != -1 :
			predictionTimesList.append(predictionTime[i])
	# print predictionTimesList

	medianPredictionTime = statistics.median(predictionTimesList)

	print "sensitivity : ", sensitivity, ", specificity : ", specificity
	print "accuracy : ", accuracy, "median prediction time : ", medianPredictionTime
	finalScore = 0
	if specificity < 0.99 or sensitivity == 0 or medianPredictionTime < 5 :
		finalScore = 0
	elif checkAllLabels(predictedData) == 1 :
		finalScore = 0
	elif checkAllPredicted(finalPrediction) == 1 :
		finalScore = 0
	elif checkAllICUFlags(predictedData) == 1 :
		finalScore = 0
	elif checkOnlyICUFlags(predictedData) == 1 :
		finalScore = 0
	else :
		sensitivityScore = sensitivity
		specificityScore = (specificity - 0.99) * 100
		if medianPredictionTime < 72 :
			medianPredictionTimeClipped = medianPredictionTime
		else :
			medianPredictionTimeClipped = 72
		medianPredictionTimeScore =  (float(medianPredictionTimeClipped)/72)
		finalScore = (75 * sensitivityScore) + (20 * medianPredictionTimeScore) + (5 * specificityScore)
	print "finalScore : ", finalScore
	return sensitivity, specificity, accuracy, medianPredictionTime, finalScore

def writecsvline(fileobj, ids, times, icus, labels):
    fileobj.write("ID,TIME,ICU,LABEL\n")
    for i in range(ids.size) :
    	fileobj.write(str(ids[i]) + "," + str(times[i]) + "," + str(icus[i]) + "," + str(labels[i]) + "\n")

def testCode() :
	originalData = pa.read_csv(
		'Training_Dataset/id_time_vitals_train.csv',
        dtype={'ID': np.int32, 'TIME': np.int32, 'ICU': np.int32}
	)

	ids = np.asarray(originalData['ID'][:1000])
	times = np.asarray(originalData['TIME'][:1000])
	icus = np.asarray(originalData['ICU'][:1000])
	labels = np.random.randint(0, 2, ids.size)
	print ids.size, times.size, icus.size, labels.size
	for i in range(ids.size) :
		if ids[i] == 5 :
			labels[i] = 0
	testFile =  open('Test.csv', 'w')
	writecsvline(testFile, ids, times, icus, labels)

def main() :
	if debug == True :
		testCode()
	validate()

if __name__ == "__main__" :
	main()