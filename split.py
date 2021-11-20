from sklearn.model_selection import train_test_split
import pandas as pd
import csv
from tqdm import tqdm

train_data = pd.read_csv('./merge.tsv', sep='\t', header=0)
for i in tqdm(range(len(train_data['tag']))):
    if train_data['tag'][i] == 0:
        train_data['tag'][i] = 'No Risk'
    elif train_data['tag'][i] == 1:
        train_data['tag'][i] = 'High Risk'


data = []
for i in range(len(train_data)):
    data.append([train_data['tag'][i], train_data['text'][i]])

x_train, x_test = train_test_split(data, test_size=0.3)
x_dev = x_test.copy()


with open('./train.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerows(x_train)

with open('./dev.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerows(x_dev)

with open('./test.tsv', 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerows(x_test)
