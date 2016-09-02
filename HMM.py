import math
import random
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd 

import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from sklearn import cross_validation as cv
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedKFold as skf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

from scipy.spatial.distance import cdist

def viterbi(O,S,p,Y,A,B) :
	T = Y.shape[0]
	T1=np.zeros((16,T))
	T2=np.zeros((16,T))
	for s in range(len(S)):
		T1[s,0] = p[s] + B.loc[s,Y[0]]
		T2[s,0] = 0
	for i in range(1,T):
		for j in range(len(S)):
			T_list=list()
			for k in range(len(S)):
				dat = (T1[k][i - 1] + A.loc[k,j] + B.loc[j,Y.loc[i]])
				T_list.append(dat)
			T1[j,i] = max(T_list)
			T2[j,i] = np.argmax(T_list)
	z=np.zeros(T,dtype=np.int8)
	z[T-1]=np.argmax(T1[:,T-1])
	x=np.zeros(T)
	x[T-1]=S[z[T-1]]
	for i in range(T-1,0,-1):
		z[i-1]=T2[z[i],i]
		x[i-1]=S[z[i-1]]
	return x

def readfile(filepath):
	name = filepath
	df = pd.read_csv(name,sep =':| ',names =["X","Y","output"],engine ='python')
	ind1 = 0
	ind2 = 200
	sequences = list()
	while ind2 <= df.shape[0]:
		df1 = df.iloc[ind1:ind2]
		df1 = df1.reset_index(drop=True)
		sequences.append(df1)
		ind1 += 201
		ind2 += 201
	return sequences

def func(df,x,coords,colors,matrix,matrix2):
	if(x.ind != 199):
		prev_loc=x.ind+1
		coord=coords[(int(x.X),int(x.Y))]
		prev_coord=coords[(int(df.iloc[prev_loc].X),int(df.iloc[prev_loc].Y))]
		matrix[prev_coord][coord]+=1
		matrix2[coord][colors[x.output]]+=1

def training(train):
	def func3(x):
		if x==0: return -999
		else: return math.log(x)
	i = 0
	j = 0
	k = 0
	coords=dict()
	colors=dict({'r':0,'g':1,'b':2,'y':3})
	for i in range(1,5):
		for j in range(1,5):
			coords[(i,j)] = k
			k += 1
	obs_probs=list()
	matrix=np.zeros((16,16))
	matrix2=np.zeros((16,4))
	priors=np.zeros(16)
	for df in train:	
		df["ind"]=df.index
		df.apply(lambda x: func(df,x,coords,colors,matrix,matrix2),axis = 1)
		priors[coords[(int(df.loc[0,"X"]),int(df.loc[0,"Y"]))]] += 1
	matpd=pd.DataFrame(matrix)
	sum1=matpd.sum(axis=1)
	sum1[sum1==0]=1
	A = matpd.loc[:,:].div(sum1, axis=0)
	S = range(0,16)
	emm = pd.DataFrame(matrix2)
	sum1 = emm.sum(axis = 1)
	sum1[sum1 == 0] = 1
	B = emm.loc[:,:].div(sum1, axis=0)
	A = A.applymap(func3)
	B = B.applymap(func3)
	O = colors.keys()
	priors = priors/200
	for p in priors:
		p = func3(p)
	return O,S,priors,A,B,colors,coords

def compare(X,Y,misses):
	acc=0
	for x,y in zip(X,Y):
		if int(x) == y:
			acc += 1
		else:
			if (int(x),y) in misses.keys():
				misses[(int(x),y)]+=1
			else:
				misses[(int(x),y)]=1
	acc = float(acc)/len(Y)
	err = 1 - acc
	return acc,err

if __name__=="__main__":
	train = readfile("train")
	test = readfile("test")
	O,S,priors,A,B,colors,coords=training(train)
	i = 0
	X = list()
	accuracies=list()
	errors=list()
	misses=dict()
	for df in test:
		print "DataFrame: " + str(i)

		Y = df["output"].apply(lambda x: colors[x])
		X = viterbi(O, S, priors, Y, A, B)

		i += 1

		yactual = df.apply(lambda x: coords[(int(x.X),int(x.Y))],axis=1)
		acc,err = compare(X,list(yactual),misses)
		accuracies.append(acc)
		errors.append(err)
	print accuracies
	print errors
	print misses
	print "MEAN ERROR: " + str(np.mean(errors))
	print "MEDIAN ERROR: " + str(np.median(errors))
	print "MEAN PERCENTAGE ACCURACY: " + str(np.mean(accuracies)*100) +"%"
