import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataInput(object):
    def __init__(self, data_dir, knn_dir, exist_mask_dir, sta_month_dir,
                 sta_feat_path, od_interfea_path,
                 k1=5, norm_opt="minmax"):
        self.train_end_time = '201812'
        self.test_end_time= '201912'
        self.train_end_idx = 6
        self.test_end_idx = 7
        self.year_od_weekend_dir = data_dir["weekend"]
        self.year_od_workday_dir = data_dir["workday"]
        self.min1, self.max1 = 0, 0
        self._mean1, self._std1 = 0, 0
        self.normmin, self.normmax = 0, 0
        self.norm_opt = norm_opt
        self.sta_month_dir = sta_month_dir
        self.sta_feat_path = sta_feat_path    
        self.od_interfea_dir = od_interfea_path
        self.exist_od_mask_dir = exist_mask_dir['od']
        self.exist_sta_mask_dir = exist_mask_dir['sta']
        self.geo_dist_knn_dir1 = knn_dir["metro"]["geo_dist"]
        self.link_dist_knn_dir1 = knn_dir["metro"]["link_dist"]
        self.func_knn_dir1 = knn_dir["metro"]["func"]
        self.trans_time_knn_dir1 = knn_dir["metro"]["transtime"]
        self.k1= k1
        self.n_sta_feat = 0

    def load_metro_demand(self):
        year_od_weekend = np.load(self.year_od_weekend_dir)
        year_od_workday = np.load(self.year_od_workday_dir)
        year_od_combined = np.stack((year_od_workday, year_od_weekend), axis=-1)
        self.min1, self.max1, self._mean1, self._std1 = year_od_combined[:self.train_end_idx+1].min(), year_od_combined[:self.train_end_idx+1].max(), year_od_combined[:self.train_end_idx+1].mean(), year_od_combined[:self.train_end_idx+1].std()
        if self.norm_opt=="minmax":
            year_od_combined_norm = self.minmax_normalize(year_od_combined, self.min1, self.max1)
        elif self.norm_opt=="std":
            year_od_combined_norm = self.std_normalize(year_od_combined, self._mean1, self._std1)
        self.normmin, self.normmax = year_od_combined_norm.min(), year_od_combined_norm.max()
        his_year_od_norm = np.concatenate((self.get_his_od(year_od_combined_norm[:,:,:,0], seq_len=3), self.get_his_od(year_od_combined_norm[:,:,:,1], seq_len=3)), axis=-1)
        return year_od_combined, year_od_combined_norm, his_year_od_norm
    
    def get_his_od(self, year_od, seq_len=3):
        '''extract the historical od data '''
        # train: 2012-2018, test: 2019
        n_years = self.test_end_idx+1
        global_n_stations = 306 # this is the global number of stations in 2024
        his_year_od = np.zeros((n_years, global_n_stations, global_n_stations, seq_len))
        for t in range(n_years):
            for i in range(seq_len):
                year_index = t - seq_len + i 
                if year_index >= 0:
                    his_year_od[t, :, :, i] = year_od[year_index]
                else:
                    continue
        return his_year_od

    def load_sta_feat(self):
        """
        Returns
        -------
        year_sta_feat : ndarray (T, N, F_s)
            Raw station features for all years.
        year_sta_feat_norm : ndarray (T, N, F_s)
            Normalized station features using train-year statistics.
        """
        year_sta_feat = np.load(self.sta_feat_path)
        exist_sta_mask = np.load(self.exist_sta_mask_dir)
        print("exist_sta_mask shape:", exist_sta_mask.shape)
        print("year_sta_feat shape:", year_sta_feat.shape)

        T, N, F_s = year_sta_feat.shape
        self.n_sta_feat = F_s
        masked_sta_features = year_sta_feat.copy()
        # Set non-existing stations to NaN so they do not affect min/max
        masked_sta_features[exist_sta_mask == 0] = np.nan
        min_vals = np.nanmin(masked_sta_features[:self.train_end_idx+1], axis=(0, 1), keepdims=True)
        max_vals = np.nanmax(masked_sta_features[:self.train_end_idx+1], axis=(0, 1), keepdims=True)
        year_sta_feat_norm = 2 * (masked_sta_features - min_vals) / (max_vals - min_vals) - 1

        return year_sta_feat, year_sta_feat_norm


    def load_od_interfea(self):
        """
        output: od_interfea, shape (T, N, N, F_i)
        """
        od_interfea=np.load(self.od_interfea_dir) 
        exist_od_mask = np.load(self.exist_od_mask_dir)
        print("exist_od_mask shape:", exist_od_mask.shape)
        print("od_interfea shape:", od_interfea.shape)
        diag_vals = np.diagonal(exist_od_mask, axis1=1, axis2=2)  # shape (T, N)
        diag_sum  = diag_vals.sum(axis=1)  # shape (T,)
        print("Number of existing self-loop OD pairs per year:", diag_sum)

        masked_od_interfea = od_interfea.copy()
        masked_od_interfea[exist_od_mask == 0] = np.nan
        min_vals = np.nanmin(masked_od_interfea[:self.train_end_idx+1], axis=(0, 1, 2), keepdims=True) 
        max_vals = np.nanmax(masked_od_interfea[:self.train_end_idx+1], axis=(0, 1, 2), keepdims=True) 
        od_interfea_norm = 2 * (masked_od_interfea - min_vals) / (max_vals - min_vals) - 1
        return od_interfea, od_interfea_norm, exist_od_mask
    
    
    def generate_od_features(self, sta_features, od_interfea):
        """
        input: sta_features: (T, N, F_s), od_interfea: (T, N, N, F_i)
        output: od_features, shape (T, N, N, F_s*2 + F_i)
        """
        T, N, F_s = sta_features.shape
        F_i = od_interfea.shape[-1]
        dim_fea = F_s*2 + F_i
        od_features = np.zeros((T, N, N, dim_fea), dtype=np.float64)
        for t in range(T):
            sta_t = sta_features[t]                        # (N,F_s)
            sta_o = sta_t[:, None, :].repeat(N, axis=1)    # (N,N,F_s)
            sta_d = sta_t[None, :, :].repeat(N, axis=0)    # (N,N,F_s)
            od_features[t] = np.concatenate(
                [sta_o, sta_d, od_interfea[t]], axis=-1
            )
        return od_features


    def compute_sample_weight(self, o_age, d_age):
        age_od = np.minimum(o_age, d_age)
        age_std = np.std(age_od[:self.train_end_idx+1])
        sample_weight = np.exp(-(age_od/age_std)**2)
        return sample_weight


    def generate_exist_new_od_idx(self):
        test_end_month_sta_filename = self.test_end_time + '.csv'
        train_end_month_sta_filename = self.train_end_time + '.csv'
        test_end_month_sta_path = os.path.join(self.sta_month_dir,test_end_month_sta_filename)
        train_end_month_sta_path = os.path.join(self.sta_month_dir,train_end_month_sta_filename)
        test_end_month_sta = pd.read_csv(test_end_month_sta_path,encoding='utf-8-sig')
        train_end_month_sta = pd.read_csv(train_end_month_sta_path,encoding='utf-8-sig')
        new_sta_list = test_end_month_sta[~test_end_month_sta['sta_id_unique'].isin(train_end_month_sta['sta_id_unique'])]['sta_id_unique'].tolist()
        print("new stations in test year:", len(new_sta_list))
        new_station_idx = np.zeros(306, dtype=int)
        for idx in new_sta_list:
            new_station_idx[idx] = 1
        new_sta_bool = new_station_idx.astype(bool)  # (306,)
        new_od_idx = new_sta_bool[:, None] | new_sta_bool[None, :]  # (306,306)
        new_od_idx_year = np.array([new_od_idx]*(self.test_end_idx+1))  # (T,306,306)

        return new_od_idx_year

    def gen_od_idx(self):
        o_idx = np.zeros((306, 306))
        d_idx = np.zeros((306, 306))
        for i in range(o_idx.shape[0]):
            o_idx[i]=i
        o_idx = np.array([o_idx]*(self.test_end_idx+1))
        for j in range(d_idx.shape[1]):
            d_idx[:,j]=j
        d_idx = np.array([d_idx]*(self.test_end_idx+1))
        return o_idx, d_idx
    
    def gen_od_age(self, sta_age):
        """sta_age: (T, 306)"""
        T = self.test_end_idx+1
        o_age = np.zeros((T, 306, 306))
        d_age = np.zeros((T, 306, 306))
        for t in range(T):
            for i in range(306):
                o_age[t, i] = sta_age[t]
                d_age[t, :, i] = sta_age[t]
        od_age = np.minimum(o_age, d_age)
        return o_age, d_age, od_age


    def load_data(self):
        print('Loading data...')
        dataset = dict()
        dataset["year_od_combined"],dataset["year_od_combined_norm"], dataset["his_year_od_norm"] = self.load_metro_demand()
        dataset["year_sta_features"],dataset["year_sta_features_norm"]=self.load_sta_feat()
        dataset["od_interfea"],dataset["od_interfea_norm"],dataset["exist_od_mask"]=self.load_od_interfea()
        dataset["od_features"] = self.generate_od_features(dataset["year_sta_features"], dataset["od_interfea"])
        dataset["od_features_norm"] = self.generate_od_features(dataset["year_sta_features_norm"], dataset["od_interfea_norm"])
        dataset["geo_dist_knn_metro"] = np.load(self.geo_dist_knn_dir1)[:, :, :self.k1+1] # T, N, k1+self
        dataset["link_dist_knn_metro"] = np.load(self.link_dist_knn_dir1)[:, :, :self.k1+1] # T, N, k1+self
        dataset["func_knn_metro"] = np.load(self.func_knn_dir1)[:, :, :self.k1+1] # T, N, k1+self
        dataset["trans_time_knn_metro"] = np.load(self.trans_time_knn_dir1)[:, :, :self.k1+1] # T, N, k1+self
        dataset['new_od_idx_year'] = self.generate_exist_new_od_idx()
        dataset['o_idx'],dataset['d_idx'] = self.gen_od_idx()
        dataset['o_sta_age'],dataset['d_sta_age'], dataset['od_age'] = self.gen_od_age(dataset['year_sta_features'][:, :, -1])
        dataset['sample_weight'] = self.compute_sample_weight(dataset['o_sta_age'], dataset['d_sta_age'])
        return dataset

    def minmax_normalize(self, x: np.array, min, max):
        x = (x - min) / (max - min)
        x = 2 * x - 1
        return x

    def minmax_denormalize(self, x: np.array, min, max):
        x = (x + 1) / 2
        x = (max - min) * x + min
        return x

    def std_normalize(self, x:np.array, mean, std):
        x = (x - mean)/std
        return x

    def std_denormalize(self, x:np.array, mean, std):
        x = x * std + mean
        return x



class DataGenerator(object):
    def __init__(self):
        self.year_len = 0
        self.train_start_time='201204'
        self.train_end_time='201812'
        self.test_end_time='201912'
        self.mode_idx = dict()
        self.train_start_idx=0
        self.train_end_idx=6
        self.test_end_idx=7


    def data2idx(self):
        '''
        return the idx for train and test
        '''
        train_idx, test_idx = [], []
        for i in range(self.train_start_idx, self.train_end_idx+1): 
                train_idx.append(i)
        for i in range(self.train_end_idx+1, self.test_end_idx+1):
                test_idx.append(i)
        return {"train": train_idx, "test": test_idx}


    def get_data_loader(self, data: dict, batch_size: int, device: str, data_class):
        feat_dict, output_dict = dict(), dict()
        output_dict["y_seq"]=data['year_od_combined_norm']
        feat_dict["year_sta_features"]=data['year_sta_features_norm']
        feat_dict["od_features"]=data['od_features']
        feat_dict["od_features_norm"]=data['od_features_norm']
        feat_dict["od_interfea"]=data['od_interfea_norm']
        feat_dict["his_year_od_norm"]=data['his_year_od_norm']
        feat_dict["geo_dist_knn_metro"] = data["geo_dist_knn_metro"]
        feat_dict["trans_time_knn_metro"] = data["trans_time_knn_metro"] 
        feat_dict["link_dist_knn_metro"] = data["link_dist_knn_metro"]
        feat_dict["func_knn_metro"] = data["func_knn_metro"]
        feat_dict['new_od_idx_year'] = data['new_od_idx_year']
        feat_dict['exist_od_mask'] = data['exist_od_mask']
        output_dict['o_sta_age'] = data['o_sta_age']
        output_dict['d_sta_age'] = data['d_sta_age']
        output_dict['od_age'] = data['od_age']
        output_dict['o_idx'] = data['o_idx']
        output_dict['d_idx'] = data['d_idx']
        feat_dict['sample_weight'] = data['sample_weight']
        
        self.mode_idx = self.data2idx() # mode_idx = {"train": train_idx, "test": test_idx}
        data_loader = dict()
        for mode in ['train', 'test']:
            samples = PrepareSample(device=device, inputs=feat_dict, output=output_dict,
                                  mode=mode, mode_idx=self.mode_idx)
            if mode == 'train':
                data_loader['train'] = DataLoader(dataset=PrepareDataset(device=device, inputs=samples.inputs, output=samples.output), \
                                               batch_size=batch_size, shuffle=True)
                data_loader['valid'] = DataLoader(dataset=PrepareDataset(device=device, inputs=samples.val_inputs, output=samples.val_output), \
                                               batch_size=batch_size, shuffle=False)
            else:
                data_loader['test'] = DataLoader(dataset=PrepareDataset(device=device, inputs=samples.inputs, output=samples.output), \
                                               batch_size=batch_size, shuffle=False)
        return data_loader



class PrepareSample(object):
    def __init__(self, device: str, inputs: dict, output: dict, mode: str, mode_idx: dict):
          self.device = device
          self.mode = mode
          self.mode_idx = mode_idx
          self.inputs, self.output, self.val_inputs, self.val_output = None, None, None, None
          self.prepare_xy(inputs, output)

    def prepare_neigh_feat(self, x_feat: np.array, knn: np.array):
        '''
        input:
        x_feat:  (T(train/test), N, n_fea)
        knn:  (T(train/test), N, k1+1)

        return:
        x_feat_neigh: (T(train/test), N, k1+1, n_fea)
        '''
        x_feat_neigh = np.zeros((knn.shape[0], knn.shape[1], knn.shape[2], x_feat.shape[-1])) # n_len, n_station, n_neigh, n_feature
        for t in range(knn.shape[0]):
            for k in range(knn.shape[-1]):
                x_feat_neigh[t, :, k, :] = x_feat[t, knn[t, :, k], :]
        return x_feat_neigh      
    
    def prepare_o_d_neigh_feat(self, feat_metro_neigh: np.array):
        """
        Parameters
        ----------
        feat_metro_neigh : ndarray, shape (T, N, K, F)
            Node-level neighbor features for each station.

        Returns
        -------
        feat_o_d_neigh : ndarray, shape (T, N, N, 2K, F)
            For each (t, o, d), concatenation of neighbors of origin and destination.
        """
        T, N, K, F = feat_metro_neigh.shape
        # Expand along destination axis for origin neighbors: (T, N, 1, K, F) -> (T, N, N, K, F)
        o_neigh = feat_metro_neigh[:, :, None, :, :].repeat(N, axis=2)
        # Expand along origin axis for destination neighbors: (T, 1, N, K, F) -> (T, N, N, K, F)
        d_neigh = feat_metro_neigh[:, None, :, :, :].repeat(N, axis=1)
        # Concatenate along neighbor dimension -> (T, N, N, 2K, F)
        feat_o_d_neigh = np.concatenate([o_neigh, d_neigh], axis=3)
        return feat_o_d_neigh

    def prepare_feat_od_interfea_neigh(self, knn: np.array, od_interfea: np.array):
        """
        Parameters
        ----------
        knn : ndarray, shape (T, N, K1)
            k-NN indices (including self) for each station at each time step.
        od_interfea : ndarray, shape (T, N, N, F_i)
            OD-level interaction features.

        Returns
        -------
        feat_od_interfea_neigh : ndarray, shape (T, N, N, K1*K1, F_i)
            For each (t, o, d), collect features for all neighbor pairs
            (o_knn[i], d_knn[j]) and flatten (i, j) into one dimension.
        """
        T, N, K1 = knn.shape
        F_i = od_interfea.shape[-1]
        n_pairs = K1 * K1

        feat_od_interfea_neigh = np.zeros(
            (T, N, N, n_pairs, F_i),
            dtype=od_interfea.dtype,
        )

        for t in range(T):
            # knn[t] -> shape (N, K1)
            o_knn = knn[t]  # origin side neighbors
            d_knn = knn[t]  # destination side neighbors

            # Build index tensors:
            # o_idx[o, d, i, j] = o_knn[o, i]
            # d_idx[o, d, i, j] = d_knn[d, j]
            o_idx = o_knn[:, None, :, None]                 # (N, 1, K1, 1)
            o_idx = np.broadcast_to(o_idx, (N, N, K1, K1))  # (N, N, K1, K1)

            d_idx = d_knn[None, :, None, :]                 # (1, N, 1, K1)
            d_idx = np.broadcast_to(d_idx, (N, N, K1, K1))  # (N, N, K1, K1)

            # Advanced indexing on od_interfea[t]: (N, N, F_i)
            # Result: (N, N, K1, K1, F_i)
            neigh_feat = od_interfea[t][o_idx, d_idx]

            # Flatten (K1, K1) -> K1*K1
            feat_od_interfea_neigh[t] = neigh_feat.reshape(N, N, n_pairs, F_i)

        return feat_od_interfea_neigh


    def prepare_o_d_knn(self, dist_knn_metro: np.array):
        """
        Parameters
        ----------
        dist_knn_metro : ndarray, shape (T, N, K)

        Returns
        -------
        knn_o_d : ndarray, shape (T, N, N, 2K)
            For each (t, o, d), concatenation of knn(o) and knn(d).
        """
        T, N, K = dist_knn_metro.shape
        # Origin neighbors: (T, N, 1, K) -> (T, N, N, K)
        knn_o = dist_knn_metro[:, :, None, :].repeat(N, axis=2)
        # Destination neighbors: (T, 1, N, K) -> (T, N, N, K)
        knn_d = dist_knn_metro[:, None, :, :].repeat(N, axis=1)
        knn_o_d = np.concatenate([knn_o, knn_d], axis=3)  # (T, N, N, 2K)
        return knn_o_d

    def prepare_xy(self, inputs: dict, output: dict):
        '''
        inputs=feat_dict, 
        output=output_dict
        '''
        print("Preparing samples for mode:", self.mode)
        print("mode idx:", self.mode_idx[self.mode])
        geo_dist_knn_metro = inputs["geo_dist_knn_metro"][self.mode_idx[self.mode]]
        link_dist_knn_metro = inputs["link_dist_knn_metro"][self.mode_idx[self.mode]] 
        time_dist_knn_metro = inputs["trans_time_knn_metro"][self.mode_idx[self.mode]]
        y_seq_metro = output['y_seq'][self.mode_idx[self.mode]]
        x_year_sta_features = inputs['year_sta_features'][self.mode_idx[self.mode]] 
        x_od_features = inputs['od_features_norm'][self.mode_idx[self.mode]] 
        x_od_interfea = inputs['od_interfea'][self.mode_idx[self.mode]] # n_len, n_station, n_station, n_od_interfea
        x_his_year_od_norm = inputs['his_year_od_norm'][self.mode_idx[self.mode]] 
        x_sample_weight = inputs['sample_weight'][self.mode_idx[self.mode]]
        feat_metro_geo_dist_neigh = self.prepare_neigh_feat(x_year_sta_features, geo_dist_knn_metro) # (50/28, 1101, n_nei=5, n_fea=44)
        feat_metro_link_dist_neigh = self.prepare_neigh_feat(x_year_sta_features, link_dist_knn_metro)
        feat_metro_time_neigh = self.prepare_neigh_feat(x_year_sta_features, time_dist_knn_metro)
        feat_metro_geo_dist_neigh_o_d = self.prepare_o_d_neigh_feat(feat_metro_geo_dist_neigh) # (50/28, 1101, n_nei=10, n_fea=44)
        feat_metro_link_dist_neigh_o_d = self.prepare_o_d_neigh_feat(feat_metro_link_dist_neigh)
        feat_metro_time_dist_neigh_o_d = self.prepare_o_d_neigh_feat(feat_metro_time_neigh)
        feat_od_interfea_neigh_geo_dist = self.prepare_feat_od_interfea_neigh(geo_dist_knn_metro, x_od_interfea)
        feat_od_interfea_neigh_link_dist = self.prepare_feat_od_interfea_neigh(link_dist_knn_metro, x_od_interfea)
        feat_od_interfea_neigh_time_dist = self.prepare_feat_od_interfea_neigh(time_dist_knn_metro, x_od_interfea)
        geo_dist_knn_metro_o_d = self.prepare_o_d_knn(geo_dist_knn_metro) 
        link_dist_knn_metro_o_d = self.prepare_o_d_knn(link_dist_knn_metro)
        time_dist_knn_metro_o_d = self.prepare_o_d_knn(time_dist_knn_metro)

        T_mode = len(self.mode_idx[self.mode])
        N = 306
        years = np.array(self.mode_idx[self.mode], dtype=np.int64) 
        y_timestep = years.reshape(T_mode, 1, 1).repeat(N, axis=1).repeat(N, axis=2)

        y_new_od_idx_year = inputs['new_od_idx_year'][self.mode_idx[self.mode]]
        y_exist_od_mask = inputs['exist_od_mask'][self.mode_idx[self.mode]]
        y_o_sta_age = output['o_sta_age'][self.mode_idx[self.mode]]
        y_d_sta_age = output['d_sta_age'][self.mode_idx[self.mode]]
        y_od_age = output['od_age'][self.mode_idx[self.mode]]
        y_o_idx = output['o_idx'][self.mode_idx[self.mode]]
        y_d_idx = output['d_idx'][self.mode_idx[self.mode]]
        y_exist_od_mask_noloop = y_exist_od_mask.copy()
        for t in range(y_exist_od_mask_noloop.shape[0]):
            np.fill_diagonal(y_exist_od_mask_noloop[t], 0)
        y_new_od_idx_year = y_new_od_idx_year.reshape(-1)
        y_exist_od_mask = y_exist_od_mask.reshape(-1)
        y_exist_od_mask_noloop = y_exist_od_mask_noloop.reshape(-1)
        exist_idx = np.where((y_new_od_idx_year == 0) * (y_exist_od_mask_noloop > 0)  > 0)[0]
        new_idx = np.where((y_new_od_idx_year > 0) * (y_exist_od_mask_noloop > 0)  > 0)[0]
        print("exist_idx", exist_idx.shape)
        print("new_idx", new_idx.shape)

        all_idx = np.concatenate([exist_idx, new_idx], 0) 
        add_label = np.concatenate([np.zeros_like(exist_idx), np.ones_like(new_idx)], 0)
        print("exist", len(exist_idx), "add", len(new_idx)) 
        print("add_label", len(add_label), np.sum(add_label)) 
       
        x_od_features = x_od_features.reshape(-1, x_od_features.shape[-1])[all_idx]
        x_his_year_od_norm = x_his_year_od_norm.reshape(-1, x_his_year_od_norm.shape[-1])[all_idx] # (n_valid_sta_month, 6)
        x_sample_weight = x_sample_weight.reshape(-1)[all_idx]
        feat_metro_geo_dist_neigh_o_d = feat_metro_geo_dist_neigh_o_d.reshape(-1, feat_metro_geo_dist_neigh_o_d.shape[-2], feat_metro_geo_dist_neigh_o_d.shape[-1])[all_idx]
        feat_metro_link_dist_neigh_o_d = feat_metro_link_dist_neigh_o_d.reshape(-1, feat_metro_link_dist_neigh_o_d.shape[-2], feat_metro_link_dist_neigh_o_d.shape[-1])[all_idx]
        feat_metro_time_dist_neigh_o_d = feat_metro_time_dist_neigh_o_d.reshape(-1, feat_metro_time_dist_neigh_o_d.shape[-2], feat_metro_time_dist_neigh_o_d.shape[-1])[all_idx]
        feat_od_interfea_neigh_geo_dist = feat_od_interfea_neigh_geo_dist.reshape(-1,feat_od_interfea_neigh_geo_dist.shape[-2],feat_od_interfea_neigh_geo_dist.shape[-1])[all_idx]
        feat_od_interfea_neigh_link_dist = feat_od_interfea_neigh_link_dist.reshape(-1,feat_od_interfea_neigh_link_dist.shape[-2],feat_od_interfea_neigh_link_dist.shape[-1])[all_idx]
        feat_od_interfea_neigh_time_dist = feat_od_interfea_neigh_time_dist.reshape(-1,feat_od_interfea_neigh_time_dist.shape[-2],feat_od_interfea_neigh_time_dist.shape[-1])[all_idx]
        geo_dist_knn_metro_o_d = geo_dist_knn_metro_o_d.reshape(-1, geo_dist_knn_metro_o_d.shape[-1])[all_idx]
        link_dist_knn_metro_o_d = link_dist_knn_metro_o_d.reshape(-1, link_dist_knn_metro_o_d.shape[-1])[all_idx]
        time_dist_knn_metro_o_d = time_dist_knn_metro_o_d.reshape(-1, time_dist_knn_metro_o_d.shape[-1])[all_idx]
        y_seq_metro = y_seq_metro.reshape(-1, y_seq_metro.shape[-1])[all_idx]
        y_o_sta_age = y_o_sta_age.reshape(-1)[all_idx]
        y_d_sta_age = y_d_sta_age.reshape(-1)[all_idx]
        y_od_age = y_od_age.reshape(-1)[all_idx]
        y_o_idx = y_o_idx.reshape(-1)[all_idx]
        y_d_idx = y_d_idx.reshape(-1)[all_idx]
        y_timestep = y_timestep.reshape(-1)[all_idx]

        if self.mode == "train":
            val_idx = np.random.choice([i for i in range(len(exist_idx))], int(len(exist_idx) * 0.2), replace=False).tolist()
            train_idx = np.ones(add_label.shape, dtype=np.int64)
            train_idx[val_idx] = 0
            train_idx = np.where(train_idx == 1)[0]
            print("val_idx", len(val_idx), "train_idx", len(train_idx)) 
 
            for mode in ["train", "valid"]:
                idx = val_idx if mode == 'valid' else train_idx 
                x = dict()
                x['x_od_features'] = torch.from_numpy(x_od_features[idx]).float().to(device)
                x["his_year_od_norm"] = torch.from_numpy(x_his_year_od_norm[idx]).float().to(device)
                x["sample_weight"] = torch.from_numpy(x_sample_weight[idx]).float().to(device)
                x['feat_metro_geo_dist_neigh_o_d'] = torch.from_numpy(feat_metro_geo_dist_neigh_o_d[idx]).float().to(device)
                x['feat_metro_link_dist_neigh_o_d'] = torch.from_numpy(feat_metro_link_dist_neigh_o_d[idx]).float().to(device)
                x['feat_metro_time_dist_neigh_o_d'] = torch.from_numpy(feat_metro_time_dist_neigh_o_d[idx]).float().to(device)
                x['timestep'] = torch.from_numpy(y_timestep[idx]).long().to(device)
                x['feat_od_interfea_neigh_geo_dist'] = torch.from_numpy(feat_od_interfea_neigh_geo_dist[idx]).float().to(device)
                x['feat_od_interfea_neigh_link_dist'] = torch.from_numpy(feat_od_interfea_neigh_link_dist[idx]).float().to(device)
                x['feat_od_interfea_neigh_time_dist'] = torch.from_numpy(feat_od_interfea_neigh_time_dist[idx]).float().to(device)
                x['geo_dist_knn_metro_o_d'] = torch.from_numpy(geo_dist_knn_metro_o_d[idx]).long().to(device)
                x['link_dist_knn_metro_o_d'] = torch.from_numpy(link_dist_knn_metro_o_d[idx]).long().to(device)
                x['time_dist_knn_metro_o_d'] = torch.from_numpy(time_dist_knn_metro_o_d[idx]).long().to(device)
                x['o_sta_age'] = y_o_sta_age[idx]
                x['d_sta_age'] = y_d_sta_age[idx]
                x['od_age'] = torch.from_numpy(y_od_age[idx]).float().to(device)
                x['o_idx'] = y_o_idx[idx]
                x['d_idx'] = y_d_idx[idx]
                x['add_label'] = torch.from_numpy(add_label[idx]).float().to(device)
                y = torch.from_numpy(y_seq_metro[idx]).float().to(device)

                if mode == "train":
                    self.inputs, self.output = x, y
                else:
                    self.val_inputs, self.val_output = x, y
        else:
            x = dict()

            x['x_od_features'] = torch.from_numpy(x_od_features).float().to(device)
            x['his_year_od_norm'] = torch.from_numpy(x_his_year_od_norm).float().to(device)
            x['sample_weight'] = torch.from_numpy(x_sample_weight).float().to(device)
            x['feat_metro_geo_dist_neigh_o_d'] = torch.from_numpy(feat_metro_geo_dist_neigh_o_d).float().to(device)
            x['feat_metro_link_dist_neigh_o_d'] = torch.from_numpy(feat_metro_link_dist_neigh_o_d).float().to(device)
            x['feat_metro_time_dist_neigh_o_d'] = torch.from_numpy(feat_metro_time_dist_neigh_o_d).float().to(device)
            x['timestep'] = torch.from_numpy(y_timestep).long().to(device)
            x['feat_od_interfea_neigh_geo_dist'] = torch.from_numpy(feat_od_interfea_neigh_geo_dist).float().to(device)
            x['feat_od_interfea_neigh_link_dist'] = torch.from_numpy(feat_od_interfea_neigh_link_dist).float().to(device)
            x['feat_od_interfea_neigh_time_dist'] = torch.from_numpy(feat_od_interfea_neigh_time_dist).float().to(device)
            x['geo_dist_knn_metro_o_d'] = torch.from_numpy(geo_dist_knn_metro_o_d).long().to(device)
            x['link_dist_knn_metro_o_d'] = torch.from_numpy(link_dist_knn_metro_o_d).long().to(device)
            x['time_dist_knn_metro_o_d'] = torch.from_numpy(time_dist_knn_metro_o_d).long().to(device)
            x['o_sta_age'] = y_o_sta_age
            x['d_sta_age'] = y_d_sta_age
            x['od_age'] = torch.from_numpy(y_od_age).float().to(device)
            x['o_idx'] = y_o_idx
            x['d_idx'] = y_d_idx
            x['add_label'] = torch.from_numpy(add_label).float().to(device)
            y = torch.from_numpy(y_seq_metro).float().to(device)
            self.inputs, self.output = x, y


class PrepareDataset(Dataset):
    def __init__(self, device: str, inputs: dict, output):
        self.device = device
        self.inputs, self.output = inputs, output

    def __len__(self):
        return self.output.shape[0]

    def __getitem__(self, item):
        for var in ['x_od_features','his_year_od_norm', 'sample_weight', 'feat_metro_geo_dist_neigh_o_d','feat_metro_link_dist_neigh_o_d',
                    'feat_metro_time_dist_neigh_o_d', 'timestep', 'feat_od_interfea_neigh_geo_dist',
                    'feat_od_interfea_neigh_link_dist','feat_od_interfea_neigh_time_dist',
                    'geo_dist_knn_metro_o_d',
                    'link_dist_knn_metro_o_d', 'time_dist_knn_metro_o_d', 'o_sta_age', 'd_sta_age', 'od_age',
                      'o_idx', 'd_idx','add_label']:
            data_element = self.inputs[var][item]
            if isinstance(data_element, np.void):
                print(var)
                print(data_element)

        return self.inputs['x_od_features'][item],\
                self.inputs['his_year_od_norm'][item],\
                self.inputs['sample_weight'][item], \
                self.inputs['feat_metro_geo_dist_neigh_o_d'][item], \
                self.inputs['feat_metro_link_dist_neigh_o_d'][item], self.inputs['feat_metro_time_dist_neigh_o_d'][item], \
                self.inputs['timestep'][item],\
                self.inputs['feat_od_interfea_neigh_geo_dist'][item], self.inputs['feat_od_interfea_neigh_link_dist'][item], \
                self.inputs['feat_od_interfea_neigh_time_dist'][item], \
                self.inputs['geo_dist_knn_metro_o_d'][item], self.inputs['link_dist_knn_metro_o_d'][item],\
                self.inputs['time_dist_knn_metro_o_d'][item], self.inputs['o_sta_age'][item], \
                self.inputs['d_sta_age'][item], self.inputs['od_age'][item], self.inputs['o_idx'][item],\
                self.inputs['d_idx'][item], \
                self.inputs['add_label'][item], \
                self.output[item] 