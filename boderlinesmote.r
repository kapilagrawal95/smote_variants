# Ref : http://dx.doi.org/10.1007/11538059_91
# see boderline smote documentaion at https://cran.r-project.org/web/packages/smotefamily/smotefamily.pdf
# Han, H., Wang, W.Y. and Mao, B.H. Borderline-SMOTE: a new over-sampling method in imbalanceddatasetslearning. 

#Usage : rscript boderlinesmote.r <raw-data-csv> <output-csv>

library(data.table)
library(smotefamily)

args = commandArgs(trailingOnly=TRUE)
d = read.csv(args[1],header = TRUE)

#change the below paramters accroding to requirment
YNcolumn = 36

print("Number of neaerest neighbour during sampling process : ")
K=as.integer(readLines("stdin",n=1))

print("Number of neaerest neighbour during safe-level calculation process : ")
C=as.integer(readLines("stdin",n=1))

print("The number or vector representing the desired times of synthetic minority instances over the original number of majority instances, 0 for duplicating until balanced : ")
dup_size = as.integer(readLines("stdin",n=1))

print("A parameter to indicate which type of Borderline-SMOTE presented in the paper is used")
print("1 for type1. 2 for type2")
M=as.integer(readLines("stdin",n=1))
if (M==1)
{
	M="type1"
} else {
	M="type2"
}


d$Class = as.numeric(d$Class)-1

D = BLSMOTE(d,d$Class,K=K,C=C,dupSize=dup_size,method=M)
D$data=D$data[,sapply(D$data,is.numeric)]

if (M=="type2")
{
	i_t=1
	i=1
	temp_indices=c()
	for (cl in D$data$Class)
	{
		if (is.na(cl))
		{
			temp_indices[i_t]=i
			i_t=i_t+1
		}
		i=i+1
	}
	D$data=D$data[-temp_indices,]
}

x=vector(mode="character",length(D$data$Class))

for (i in seq(1,length(D$data$Class))) {
	if (D$data$Class[i]==1)
	{
		x[i]="Y"
	} else {
		x[i]="N"
	}
}
D$data$Class=x
fwrite(D$data,args[2])
