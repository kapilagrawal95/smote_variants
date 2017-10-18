library(DMwR)
library(data.table)

args = commandArgs(trailingOnly=TRUE)
d = read.csv(args[1],header = TRUE)

#change the below paramters accroding to requirment
YNcolumn = 36

print("Specify k as in knn of SMOTE : ")
k_knn=as.integer(readLines("stdin",n=1))

countyes <- function(dataset)
{
	y=0
	for (cl in dataset[,YNcolumn])
	{
		if (cl=="Y")
			y=y+1
	}
	return (y)
}

y=countyes(d)
ov = ((length(d[,YNcolumn])-y)/y)*100
un = (ov*y/100)*100/(length(d[,YNcolumn])-(2*y))	#please ignore these
print("Original Data")
print(table(d$Class))

D = SMOTE(Class~.,d,perc.over=ov,perc.under=un,k=k_knn,learner=NULL)	#apply SMOTE on original data
print("SMOTEd Data")
print(table(D$Class))

fwrite(D,args[2])	#write final dataset to another csv
