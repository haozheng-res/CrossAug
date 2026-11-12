import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


warnings.filterwarnings('ignore')


class ExogenousDatasetBenchmark(Dataset):
    """Exogenous dataset with multivariate inputs and univariate outputs.
    seq_x contains all variables, with the target always in the last column; seq_y contains only the target.
    MS: The last column is the fixed target.
    M:  Each column takes a turn as the target and is dynamically moved to the last column.
    """
    def __init__(self, args, root_path, data_path, flag, size, features='M'):
        self.args = args
        self.seq_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        assert self.args.data in ['custom', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Solar']
        assert self.args.features in ['M', 'MS']

        self.features = features

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset_file_path = os.path.join(self.root_path, self.data_path)
        if dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)
        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]  # Select only column 0 / feature 0.
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        if df_raw.isnull().values.any():
            raise ValueError(f"NaN found in {self.data_path}, please clean the data first.")
        print(f'df_raw.shape: {df_raw.shape}')

        # Dataset split boundaries.
        if self.args.data == 'custom':
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]
        elif self.args.data in ('ETTh1', 'ETTh2'):
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.args.data in ('ETTm1', 'ETTm2'):
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        elif self.args.data == 'Solar':
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = int(data_len * 0.1)
            border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Extract numeric columns (skip the date column).
        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        n_total_var = data.shape[1]

        # Standardization: fit on the training set.
        train_data = data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        data = self.scaler.transform(data)

        self.data_x = data[border1:border2]  # (T, n_total_var)
        self.data_y = data[border1:border2]  # (T, n_total_var)

        self.n_var = 1 if self.features == 'MS' else n_total_var
        self.n_timepoint = len(self.data_x) - self.seq_len - self.pred_len + 1
        self.samples_per_var = max(int(self.n_timepoint * self.args.subset_sampling_ratio), 1)
        # Caching additionally exposes stable global dataset indices only for train/val; default and test return values stay unchanged.
        self.return_sample_index = (self.args.cache_ltm_outputs and self.set_type in (0, 1))

    def __getitem__(self, index):
        if self.set_type == 0:  # Training set: uniformly spaced sparse sampling.
            c_begin = index // self.samples_per_var
            local_idx = index % self.samples_per_var
            interval = max(self.n_timepoint // self.samples_per_var, 1)
            s_begin = local_idx * interval
        else:
            c_begin = index // self.n_timepoint
            s_begin = index % self.n_timepoint

        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        if self.features == 'MS':
            seq_x = self.data_x[s_begin:s_end]                               # (seq_len, n_total_var), with the target in the last column.
            seq_y = self.data_y[r_begin:r_end, -1:]                           # (pred_len, 1)
        else:
            # Move the target column to the end, preceded by exogenous variables.
            raw = self.data_x[s_begin:s_end]                                  # (seq_len, n_total_var)
            exog_idx = [i for i in range(raw.shape[1]) if i != c_begin]
            seq_x = np.concatenate([raw[:, exog_idx], raw[:, c_begin:c_begin+1]], axis=1)  # (seq_len, n_total_var)
            seq_y = self.data_y[r_begin:r_end, c_begin:c_begin+1]            # (pred_len, 1)

        if self.return_sample_index:
            return seq_x, seq_y, index
        return seq_x, seq_y

    def __len__(self):
        if self.set_type == 0:
            return self.n_var * self.samples_per_var
        else:
            return self.n_var * self.n_timepoint
