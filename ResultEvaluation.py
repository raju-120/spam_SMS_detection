import pandas as pd

df = pd.read_csv("./data/spam.csv", encoding='latin-1')
# Dropping unwanted columns
df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True, axis=1)

# Changing the labels for convinience
df["v1"].replace({"ham": 0, "spam":1}, inplace=True)

# Changing the column names for better
df.rename({"v1": "spam", "v2": "original_message"},axis=1, inplace=True)


out = pd.read_csv("./output/spam_encoded.csv")

#out.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True, axis=1)

#print(len(df["spam"]))
#print(len(out["spam"]))
tp = 0 #true_positive
tn = 0 #true_negative
fp = 0 #false_positive
fn =0 #false_negative
for x in range(len(df["spam"])):
    if(df["spam"][x]==0 and out["spam"][x]==0):
        tn +=1
    if (df["spam"][x] == 0 and out["spam"][x] == 1):
        fp += 1
    if (df["spam"][x] == 1 and out["spam"][x] == 1):
        tp += 1
    if (df["spam"][x] == 1 and out["spam"][x] == 0):
        fn += 1



accuracy = (tp+ tn)/ (tn+ fp+ fn + tp)
precision = tp/ (tp+fp)
recall = tp/tp+fn
F1 = 2*(recall* precision)/(recall+ precision)

print("Accuracy: " + str(accuracy * 100) + " % ")
print("Precision: " +str(precision))
print("Recall: " +str(recall))
print("F1 Score: " +str(F1))

