#input file as first argument
#output file as second argument

#requires python 3.5+
#needs numpy, scipy, imblearn and scikit-learn modules installed
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn import tree
import random
import sys
from os import urandom
from binascii import hexlify
import numpy as np

def shuffle_together(a,b):	#function to shuffle 2 numpy arrays randomly and with sync
	seed = int(hexlify(urandom(2)),16)
	random.seed(seed)
	assert len(a)==len(b)
	sa = np.empty(shape=a.shape,dtype=a.dtype)
	sb = np.empty(shape=b.shape,dtype=b.dtype)
	indexing = [i for i in range(0,len(a))]
	for i in range(0,10):
		random.shuffle(indexing)
	for old_i,new_i in enumerate(indexing):
		sa[new_i]=a[old_i]
		sb[new_i]=b[old_i]
	return sa,sb

#change parameters below according to requirments
YNcolumn = 35	#column containing Y/N. This column should be last column of csv file.
heading_present = 1	# 1 if heading is present in csv file, 0 if not present.

n = int(input('\nNumber of sub-datsets to make : '))	# number of parts into which dataset is to be divided for decision tree classifier
k = int(input('\nNumber of iterations to identify noisy examples : '))	# no of iterations to identify noisy examples
voting = int(input('\n1 for majority, 0 for consensus : '))	# 1 for majority, 0 for consensus
p = float(input('\npercentage for original size to consider : '))

fr = open(sys.argv[1])
if heading_present:
	heading = fr.readline()

dat = []
yn = []

for r in fr.readlines():
	dat.append([float(e) for e in r[:-3].split(',')])
	yn.append(r[-2])

fr.close()

d = np.array(dat,dtype = np.float32)	#convert list into numpy array
y = np.array(yn,dtype = np.dtype('U2'))
sm = SMOTE(random_state=42)	#apply SMOTE on it
#Ref : http://contrib.scikit-learn.org/imbalanced-learn/generated/imblearn.over_sampling.SMOTE.html
D,Y = sm.fit_sample(dat,yn)	#obtaining SMOTEd dataset

print('\nOriginal Dataset : ',Counter(y))
P = int((len(d)*p)/100)	#maximum no of noisy examples that can be tolerated

i_k=0
#looping for k starts
while i_k<k:
	L=len(D)
	step = int(L/n)	#no of elements in each dataset except last one
	models = []	#list to store pridection models created by DecsionTreeClassifier
	D,Y=shuffle_together(D,Y)	#shuffle both list randomly and synchronously so each sub-datset contains roughly equal instances of both class

	#making n decision models by splitting the data into n parts
	for i in range(0,n-1):
		clf = tree.DecisionTreeClassifier()
		models.append(clf.fit(D[i*step:(i+1)*step],Y[i*step:(i+1)*step]))	#making n-1 models
	clf = tree.DecisionTreeClassifier()
	models.append(clf.fit(D[(n-1)*step:],Y[(n-1)*step:]))	#making n-th model

	predictions = []
	for i in range(0,n):
		predictions.append(models[i].predict(D))	#storing the preidiction of whole dataset made by each model

	temp_indices = []	#stores the indices of data which has to removed
	if voting:
		for i in range(0,L):
			positive = 0
			negetive = 0
			for j in range(0,n):
				if Y[i]==predictions[j][i]:
					positive += 1
				else:
					negetive += 1
			if positive<negetive:
				temp_indices.append(i)
	else:
		for i in range(0,L):
			misclassified_by_all = 1
			for j in range(0,L):
				if Y[i]==predictions[j][i]:
					misclassified_by_all = 0
					break
			if misclassified_by_all:
				temp_indices.append(i)

	#remove noisy examples from main dataset
	D = np.delete(D,temp_indices,0)	#removing data
	Y = np.delete(Y,temp_indices)
	if len(temp_indices)<=P:
		i_k += 1	#stoping criterion is reached only when no. of examples is less than P for three times consequtively
	else:
		i_k = 0		#if condition is not met counter again set to 0

print('Preprocessed Dataset : ',Counter(Y))	#prints no of 'Y' and 'N' examples

fw=open(sys.argv[2],'w')	#file for output
if heading_present:
	fw.write(heading)

for i,r in enumerate(D):	#write in csv format
	for s in r:
		fw.write(str(s)+',')
	fw.write(Y[i]+'\n')

fw.close()