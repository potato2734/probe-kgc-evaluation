import json
import os

datas = os.listdir('./')
datas.remove('adjust.py')

for data in datas:
    info_ls = []

    files = os.listdir(data)
    for file in files:
        with open(os.path.join(data, file), 'r') as f:
            info_ls.append(json.load(f))
    for file, info in zip(files, info_ls):
        new_info_ls = []
        for ls in info:
            q,m,r,qc,qe,f=ls
            new_info_ls.append([q,m,r,qc,qe,f-1])
        with open(os.path.join(data, file), 'w') as f:
            json.dump(new_info_ls, f, indent=4)

