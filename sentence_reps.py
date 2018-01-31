import random
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
import numpy as np
import data

convert = {
    'government': 0,
    'fiction': 1,
    'slate': 2,
    'telephone': 3,
    'travel': 4,
    'letters': 5,
    'facetoface': 6,
    'nineeleven': 7,
    'oup': 8,
    'verbatim': 9
}

_, _, _, matched_genres = data.load_snli('multinli/dev_matched.txt')
print(matched_genres)
_, _, _, mismatched_genres = data.load_snli('multinli/dev_mismatched.txt')
matched_genres = [convert[g] for g in matched_genres]
mismatched_genres = [convert[g] for g in mismatched_genres]
hx_matched = np.load('multinli/h_dev_matched.npy')
hx_mismatched = np.load('multinli/h_dev_mismatched.npy')
# hx_matched = np.load('multinli/hx_dev_matched.npy')
# hx_mismatched = np.load('multinli/hx_dev_mismatched.npy')
# hy_matched = np.load('multinli/hy_dev_matched.npy')
# hy_mismatched = np.load('multinli/hy_dev_mismatched.npy')
# h1 = zip(hx_matched.tolist(), matched_genres)
h1 = zip(hx_matched.tolist(), matched_genres)
# h2 = zip(hy_matched.tolist(), matched_genres)
# h3 = zip(hx_matched.tolist() + hy_matched.tolist(), matched_genres + matched_genres)
# h4 = zip(hx_mismatched.tolist(), mismatched_genres)
h4 = zip(hx_mismatched.tolist(), mismatched_genres)
# h5 = zip(hy_mismatched.tolist(), mismatched_genres)
# h6 = zip(hx_mismatched.tolist() + hy_mismatched.tolist(), mismatched_genres + mismatched_genres)

for ho in (h1, h4):
    random.shuffle(ho)
    h, o = zip(*ho)
    h = [np.array(l) for l in h]
    clf = linear_model.LogisticRegression()
    pred = cross_val_predict(clf, np.array(h), o, cv=5)
    correct = 0
    if o[0] <= 4:
        labels = [0, 1, 2, 3, 4]
    else:
        labels = [5, 6, 7, 8, 9]
    for i in labels:
        tot = 0.0
        for k in range(len(o)):
            if o[k] == i:
                tot += 1
        for j in labels:
            count = 0.0
            for k in range(len(o)):
                if pred[k] == j and o[k] == i:
                    count += 100
                    if i == j:
                        correct += 1
            print '{:5.2f}\\% {}'.format(count / len(o), '\\\n' if j == 4 or j == 9 else '& '),
    print(correct)
