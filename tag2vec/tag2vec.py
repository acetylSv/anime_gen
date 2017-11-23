import sys
'''
# for collect tags' distribution
tag_dict = {}
with open(sys.argv[1]) as f:
    for line in f:
        idx, tags = line.strip().split(',')
        tags = tags.split('\t')
        for t in tags:
            t = t.split(':')[0].strip()
            if t not in tag_dict:
                tag_dict[t] = 1
            else:
                tag_dict[t] += 1

tag_dict_list = sorted((value,key) for (key,value) in tag_dict.items())
for t in tag_dict_list:
    print(t)
'''

import numpy as np

attr_dict = {
    'orange hair':1, 'white hair':2, 'aqua hair':3, 'gray hair':4,
    'green hair':5, 'red hair':6, 'purple hair':7, 'pink hair':8,
    'blue hair':9, 'black hair':10, 'brown hair':11, 'blonde hair':12,
    'no_hair_color':13,

    'short hair':14, 'long hair':15,
    'no_hair_length':16,

    'gray eyes':17, 'bicolored eyes':18, 'black eyes':19, 'orange eyes':20,
    'pink eyes':21, 'yellow eyes':22, 'aqua eyes':23, 'purple eyes':24,
    'green eyes':25, 'brown eyes':26, 'red eyes':27, 'blue eyes':28,
    'no_eyes_color':29
}

tag2vec = []
with open(sys.argv[1]) as f:
    for line in f:
        temp = np.zeros(29)
        idx, tags = line.strip().split(',')
        tags = tags.split('\t')
        for t in tags:
            t = t.split(':')[0].strip()
            if t in attr_dict:
                temp[attr_dict[t]-1] = 1
        if np.sum(temp[0:13]) == 0:
            temp[12] = 1
        if np.sum(temp[13:16]) == 0:
            temp[15] = 1
        if np.sum(temp[16:29]) == 0:
            temp[28] = 1
        tag2vec.append(temp)

np.save('tag_vec.npy', np.array(tag2vec))
