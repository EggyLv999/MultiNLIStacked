#this code is pretty horrible

import math

c = 'contradiction'
e = 'entailment'
n = 'neutral'

f = open('bilstm_dev_matched.txt')
tags = f.readline()
print(tags)
agreed = 0.0
disagreed = 0.0
agreed_incorrect = 0.0
disagreed_incorrect = 0.0
correct_entropy = 0.0
incorrect_entropy = 0.0
correct = 0.0
incorrect = 0.0
for line in f:
    cols = line.split('\t')
    # kill newline
    cols[-1] = cols[-1][:-1]
    if cols[-1] == cols[-2] and cols[-2] == cols[-3] and cols[-3] == cols[-4] and cols[-4] == cols[-5]:
        agreed += 1
        if cols[0] != cols[1]:
            agreed_incorrect += 1.0
            incorrect += 1
        else:
            correct += 1
    else:
        disagreed += 1
        if cols[0] != cols[1]:
            incorrect += 1
            disagreed_incorrect += 1.0
        else:
            correct += 1
    es = 0.0
    cs = 0.0
    ns = 0.0
    for tag in cols[-5:]:
        if tag == e:
            es += 1
        if tag == c:
            cs += 1
        if tag == n:
            ns += 1
    entropy = -sum([0 if x == 0 else (x/5) * math.log(x/5, 2) for x in [es, cs, ns]])
    if cols[0] != cols[1]:
        incorrect_entropy += entropy
    else:
        correct_entropy += entropy

print(agreed_incorrect / agreed)
print(disagreed_incorrect / disagreed)
print(agreed)
print(disagreed)
print(correct_entropy / correct)
print(incorrect_entropy / incorrect)
