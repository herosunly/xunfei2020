# coding=utf-8


import os
import pandas as pd
from conf import config
import json

if __name__ == '__main__':
    bad_files = pd.read_csv('bad_files.csv')
    print(bad_files.values)

    classes = [filename for filename in os.listdir(config.wav_train_path)]
    label_map = dict(zip(classes, [i for i in range(config.num_classes)]))
    print(classes, label_map)

    filename = []
    label = []
    for c in classes:
        for t in ['train', 'dev']:
            tmp_filename = os.listdir(os.path.join(config.wav_train_path, c, t))
            tmp_filename = [os.path.join(config.wav_train_path, c, t, f) for f in tmp_filename]
            for f in tmp_filename:
                if f in bad_files.values:
                    continue
                filename.append(f)
                label.append(label_map[c])
    train_df = pd.DataFrame()
    train_df['filename'] = filename
    train_df['label'] = label

    json_str = json.dumps(label_map)
    with open(os.path.join(config.data_path, 'label_map.json'), 'w') as f:
        f.write(json_str)

    train_df.to_csv(os.path.join(config.data_path, 'train.csv'), index=None)
