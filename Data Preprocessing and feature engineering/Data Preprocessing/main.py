import pandas as pd

with open("iris.data") as f:
    data = f.read()
    data = data.split("\n")
    
NewData = []
for line in data:
    NewData.append(line.split(","))
    
# print(NewData[0][4])

df = pd.DataFrame(NewData, columns =["C1","C2","C3","C4","Type"])

df.to_excel("iris.xlsx", index = False)