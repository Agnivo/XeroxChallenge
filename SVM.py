from library import lasagnennet as LN
from library import features as F
from library import scale as S
from library import validate as V
from sklearn import svm
from sklearn.metrics import roc_curve

import numpy as np
import theano as TH
import pandas as pa
import lasagne as L


def main():
    train_feats = pa.read_csv('train_feats.csv')
    train_targets = pa.read_csv('train_targets.csv')
    val_feats = pa.read_csv('validation_feats.csv')
    val_targets = pa.read_csv('validation_targets.csv')

    tx = np.asarray(train_feats, dtype=TH.config.floatX)
    tyall = np.asarray(train_targets, dtype=TH.config.floatX)

    ty = np.asarray(
        train_targets['LABEL'], dtype=TH.config.floatX).reshape(-1, 1)
    vx = np.asarray(val_feats, dtype=TH.config.floatX)
    vyall = np.asarray(val_targets, dtype=TH.config.floatX)
    vy = np.asarray(
        val_targets['LABEL'], dtype=TH.config.floatX).reshape(-1, 1)

    ty = ty.ravel()
    vy = vy.ravel()

    clf = svm.SVC(verbose=True, max_iter=100, class_weight={0: 1, 1: 3})
    clf.fit(tx, ty)

    probs = clf.predict_proba(vx)

    yscore = probs[:, 1]
    fpr, tpr, thresh = roc_curve(vy, yscore)

    thresh = np.percentile(probs[:, 1], 99)
    pred = np.int32(probs[:, 1] >= thresh)
    # for i in xrange(len(pred)):
    #     id = np.int32(vyall[i,0])
    #     if idtimes[id]-vyall[i,1] < 5*60*60:
    #         pred[i] = 0
    print np.sum(pred)

    output = vyall.swapaxes(0, 1)
    output[2] = pred
    output = output.swapaxes(0, 1)

    outfile = open('out.csv', 'w')
    outfile.write('ID,TIME,LABEL,ICU\n')
    for i in output:
        if i[3] == 1:
            outfile.write(','.join([str(int(j)) for j in i]) + '\n')
    outfile.close()

    V.validate('out.csv', './Training_Dataset/id_label_train.csv')

if __name__ == '__main__':
    main()
