import numpy as np
import json
import csv


with open('training10b.json') as j1:
    data1 = json.load(j1)
j1.close()



fact = "factoid"
yn = "yesno"
yy = "yes"
nn = "no"
lxt = "list"
summ = "summary"

factoid = []
yes = []
no = []
lst = []
summary = []
missed = []

num1 = 0
for question in data1["questions"]:
    num1 += 1
    if question['type'] == fact:
        factoid.append(question['body'])
    elif question['type'] == yn and question['exact_answer'] == yy:
        yes.append(question['body'])
    elif question['type'] == yn and question['exact_answer'] == nn:
        no.append(question['body'])
    elif question['type'] == lxt:
        lst.append(question['body'])
    elif question['type'] == summ:
        summary.append(question['body'])
    else:
        missed.append(question['body'])

print(num1)

factoidArr = np.array(factoid)
yesArr = np.array(yes)
noArr = np.array(no)
listArr = np.array(lst)
summaryArr = np.array(summary)


fSize = len(factoidArr)
ynSize = len(yesArr) + len(noArr)
lSize = len(listArr)
smSize = len(summaryArr)

#print(fSize)
#print(ynSize)
#print(lSize)
#print(smSize)

z1 = np.zeros([fSize, 4], dtype=int)
z2 = np.zeros([ynSize, 4], dtype=int)
z3 = np.zeros([lSize, 4], dtype=int)
z4 = np.zeros([smSize, 4], dtype=int)
z1[:, 0] = 1
z2[:, 1] = 1
z3[:, 2] = 1
z4[:, 3] = 1
conX = np.concatenate((factoidArr, yesArr, noArr, listArr, summaryArr))
conY = np.concatenate((z1, z2, z3, z4))

#print(conX)
#print(conY)

header = ["Text", "Label"]
k = zip(conX, conY)
with open("nlp.csv", "w") as newfile:
    csv_writer = csv.writer(newfile, delimiter="\t", lineterminator="\n")
    csv_writer.writerow(header)
    for x in k:
        csv_writer.writerows(k)
newfile.close()

