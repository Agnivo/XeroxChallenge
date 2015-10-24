import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# def interpolate():
    # vitals = pa.read_csv(
    #     'Training_Dataset/id_time_vitals_train.csv',
    #     dtype={'ID': np.int32, 'TIME': np.int32, 'ICU': np.int32}
    # )
    # labs = pa.read_csv(
    #     'Training_Dataset/id_time_labs_train.csv',
    #     dtype={'ID': np.int32, 'TIME': np.int32}
    # )
    # ages = pa.read_csv(
    #     'Training_Dataset/id_age_train.csv',
    #     dtype={'ID': np.int32, 'AGE': np.int32}
    # )
values = [np.nan, np.nan, 329285, 260260, np.nan, np.nan]
times = ['31500','40000','50000','52000','55000','60000']
timestamps = [datetime.datetime.fromtimestamp(float(x)) for x in times]

ts = pd.Series(values, index=timestamps)
# for i, row in ts.iteritems():
#     if np.isnan(row):
#     	row = 21752
#     print i,row
ts = ts.fillna(value=21752, limit=1)
ts[ts==-1] = np.nan
print ts
# ts.loc[datetime.datetime.fromtimestamp(float('60000'))] = 31258
# print ts
ts = ts.interpolate(method='spline', order=2)
print ts
print ts.loc[datetime.datetime.fromtimestamp(float('40000'))]
ts = ts.resample('60T', how='mean')
# print ts
# print ts.interpolate(method='spline', order=3)
ts = ts.interpolate(method='time')
# for i, row in ts.iteritems():
	# print i.hour," : ", row
# lines, labels = plt.gca().get_legend_handles_labels()
# labels = ['spline', 'time']
# plt.legend(lines, labels, loc='best')
# plt.show()
    

# def main():
#     interpolate()


# if __name__ == "__main__":
#     main()
