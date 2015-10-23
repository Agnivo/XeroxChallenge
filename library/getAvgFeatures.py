import sys
import pandas as pd
import math

if __name__ == '__main__':
	train_vitals = pd.read_csv(sys.argv[1])
	avg_vitals = {}
	count_vitals = {}
	key_vitals = train_vitals.keys()
	
	num = len(train_vitals[key_vitals[0]])

	for i in xrange(0,num):
		idd = int(train_vitals['ID'][i])
		time = int(train_vitals['TIME'][i])/3600 
		if idd not in avg_vitals:
			avg_vitals[idd] = {}
			count_vitals[idd] = {}
		
		if time not in avg_vitals[idd]:
			avg_vitals[idd][time] = {}
			count_vitals[idd][time] = 0
		
		count_vitals[idd][time] += 1
		
		for key in key_vitals[2:]:
			if key not in avg_vitals[idd][time]:
				avg_vitals[idd][time][key] = 0.0
			val = float(train_vitals[key][i])
			if not math.isnan(val):
				avg_vitals[idd][time][key] += val
	
	for idd,udict in avg_vitals.items():
		for time,tdict in udict.items():
			for key,val in tdict.items():
				avg_vitals[idd][time][key] /= count_vitals[idd][time]

	with open(sys.argv[2], 'w') as f:
		f.write(key_vitals[0])
		for key in key_vitals[1:]:
			f.write(',' + key)
		f.write('\n')
		for idd,udict in avg_vitals.items():
			for time,tdict in udict.items():
				f.write(str(idd)+','+str(time))
				for key,val in tdict.items():
					f.write(',' + str(val) )
				f.write('\n')