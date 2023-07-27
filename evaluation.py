import fastwer

f = open("ocr_v2/pred.txt", "r")
g = open("ocr_v2/gt.txt", "r")
preds = []
labels = []

lines = f.readlines()
for line in lines:
    txt = line.strip().split('\t')[1]
    preds.append(txt)

lines = g.readlines()
for line in lines:
    txt = line.strip().split('\t')[1]
    labels.append(txt)

cer = fastwer.score(preds, labels, char_level=True)
wer = fastwer.score(preds, labels, char_level=False)

err = 0
for i in range(len(preds)):
    if preds[i] != labels[i]:
        err+=1

print(cer)
print(wer)
print(1.00-err*1.0/len(preds))
