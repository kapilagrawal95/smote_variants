#SMOTE-ENN docementation : http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.combine.SMOTEENN.html#imblearn.combine.SMOTEENN

#Usage : python3 smoteenn.py <raw-data-csv> <output-csv>

import numpy as np
import sys
from random import random
from imblearn.combine import SMOTEENN

f=open(sys.argv[1])
k = int(input('specify value of k for KNN : '))
m = int(input('Number of nearest neighbours to use to determine if a minority sample is in danger : '))

#change below parameters according to requirment
indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
#numbers in indices list should represent columns of float type
YNcolumn = 35	#column containing Y/N. This column should be last column of csv file.
heading_present = 1	# 1 if heading is present in csv file, 0 if not present.

if heading_present:
	heading=f.readline()

d=[]
yn=[]
for i,r in enumerate(f.readlines()):
	d.append(r[:-1].split(','))
	yn.append(d[i].pop())
	for j,atom in enumerate(d[i]):
		d[i][j]=float(atom)

f.close()

d_resampled1,yn_resampled1 = SMOTEENN(k=k,m=m).fit_sample(d,yn)

def filewrite(outfile,X,y):
	if heading_present:
		outfile.write(heading)
	for i,x in enumerate(X):
		for j in indices:
			outfile.write(str(x[j])+',')
		outfile.write(y[i]+'\n')
	outfile.close()

filewrite(open(sys.argv[2],'w'),d_resampled1,yn_resampled1)
