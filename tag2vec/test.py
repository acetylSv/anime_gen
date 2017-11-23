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

inv_map = {v: k for k, v in attr_dict.items()}

a = np.load(open('tag_vec.npy', 'rb'))
for l in a:
    print('====')
    for idx, i in enumerate(l):
        if i == 1:
            print(inv_map[idx+1])
