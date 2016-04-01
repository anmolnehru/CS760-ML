import sys

table [[]]
#attribute []
arr[]

f=open("lymph_train.arff")
count=0 
for line in f:
	count+=1
        #line=line.strip("\n")
        #line=line.split(',')
        if not (line.startswith("%")):
                if not (line.startswith("@")):
                        line=line.strip("\n")
                        line=line.split(',')
                        arr.append(line)
			print(line)

		else:
			//split logic


#probability and count based logic

			
