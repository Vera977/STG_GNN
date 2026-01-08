import os
import datetime
import numpy as np
import torch
import torch.nn as nn


class ModelTrainer(object):
    def __init__(self, model_name: str, model:nn.Module, loss:nn.Module, optimizer, lr:float, wd:float, n_epochs:int, data_class, agew):
        self.model = model
        self.model_name = model_name
        self.criterion = loss
        self.cluster_criterion = nn.NLLLoss()
        self.optimizer = optimizer(params=self.model.parameters(), lr=lr, weight_decay=wd)
        self.n_epochs = n_epochs
        self.data_class = data_class
        self.agew = agew 

    @staticmethod
    def _log_non_finite(tensor: torch.Tensor, name: str):
        if not torch.is_tensor(tensor):
            return
        finite_mask = torch.isfinite(tensor)
        if not finite_mask.all():
            num_bad = tensor.numel() - finite_mask.sum().item()
            print(f"[WARN] Non-finite values in {name}: {num_bad}/{tensor.numel()}")

    def get_batch_feat_neigh_od(
        self,
        batch_feat_neigh_o_d: torch.Tensor,
        batch_feat_od_interfea_neigh: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine station-level neighbor features (origin & destination)
        with OD-level neighbor interaction features for all neighbor pairs.

        Parameters
        ----------
        batch_feat_neigh_o_d : Tensor, shape (B, 2*K1, F_s)
            For each OD pair in the batch, concatenated neighbor features of
            origin station and destination station:
                - first K1: origin neighbors
                - next  K1: destination neighbors
        batch_feat_od_interfea_neigh : Tensor, shape (B, K1*K1, F_i)
            OD interaction features for all neighbor pairs (i, j)

        Returns
        -------
        batch_feat_neigh_od : Tensor, shape (B, K1*K1 - 1, 2*F_s + F_i)
            For each OD pair in the batch, concatenated features
            [o_neigh_feat(i), d_neigh_feat(j), od_interfea_neigh(i,j)]
            for all neighbor pairs, excluding the first pair (the self-self pair).
        """
        B, twoK1, F_s = batch_feat_neigh_o_d.shape
        K1 = twoK1 // 2
        F_i = batch_feat_od_interfea_neigh.shape[-1]

        feat_neigh_o = batch_feat_neigh_o_d[:, :K1, :]   # (B, K1, F_s)
        feat_neigh_d = batch_feat_neigh_o_d[:, K1:, :]   # (B, K1, F_s)

        device = batch_feat_neigh_o_d.device
        idx = torch.arange(K1, device=device)

        idx_o = idx.repeat_interleave(K1)  
        idx_d = idx.repeat(K1)     

        feat_o = feat_neigh_o[:, idx_o, :]  # (B, K1*K1, F_s)
        feat_d = feat_neigh_d[:, idx_d, :]  # (B, K1*K1, F_s)

        neigh = torch.cat([feat_o, feat_d, batch_feat_od_interfea_neigh], dim=-1)  # (B, K1*K1, 2*F_s + F_i)
        neigh = neigh[:, 1:, :]  # (B, K1*K1 - 1, 2*F_s + F_i)

        return neigh


    def train(self, data_processor:dict, modes:list, model_dir:str, data_class, early_stopper=10):
        checkpoint = {'epoch':0, 'state_dict':self.model.state_dict()}
        val_loss = np.inf
        start_time = datetime.datetime.now()
        # print(start_time)
        train_loss, valid_loss = [], []  

        for mode in modes:
            if mode not in data_processor:
                continue
            loader = data_processor[mode]
            try:
                batch = next(iter(loader))
            except StopIteration:
                continue
            (x_od_features, his_year_od_norm, sample_weight, feat_metro_geo_dist_neigh_o_d,
             feat_metro_link_dist_neigh_o_d, feat_metro_time_dist_neigh_o_d, timestep,
             feat_od_interfea_neigh_geo_dist, feat_od_interfea_neigh_link_dist,
             feat_od_interfea_neigh_time_dist, geo_dist_knn_metro_o_d, link_dist_knn_metro_o_d,
             time_dist_knn_metro_o_d, o_sta_age, d_sta_age, od_age, o_idx, d_idx, add_label,
             y_true) = batch
            print(f"[INFO] NaN/Inf check on first batch ({mode})")
            self._log_non_finite(x_od_features, "x_od_features")
            self._log_non_finite(his_year_od_norm, "his_year_od_norm")
            self._log_non_finite(sample_weight, "sample_weight")
            self._log_non_finite(feat_metro_geo_dist_neigh_o_d, "feat_metro_geo_dist_neigh_o_d")
            self._log_non_finite(feat_metro_link_dist_neigh_o_d, "feat_metro_link_dist_neigh_o_d")
            self._log_non_finite(feat_metro_time_dist_neigh_o_d, "feat_metro_time_dist_neigh_o_d")
            self._log_non_finite(timestep, "timestep")
            self._log_non_finite(feat_od_interfea_neigh_geo_dist, "feat_od_interfea_neigh_geo_dist")
            self._log_non_finite(feat_od_interfea_neigh_link_dist, "feat_od_interfea_neigh_link_dist")
            self._log_non_finite(feat_od_interfea_neigh_time_dist, "feat_od_interfea_neigh_time_dist")
            self._log_non_finite(geo_dist_knn_metro_o_d, "geo_dist_knn_metro_o_d")
            self._log_non_finite(link_dist_knn_metro_o_d, "link_dist_knn_metro_o_d")
            self._log_non_finite(time_dist_knn_metro_o_d, "time_dist_knn_metro_o_d")
            self._log_non_finite(o_sta_age, "o_sta_age")
            self._log_non_finite(d_sta_age, "d_sta_age")
            self._log_non_finite(od_age, "od_age")
            self._log_non_finite(o_idx, "o_idx")
            self._log_non_finite(d_idx, "d_idx")
            self._log_non_finite(add_label, "add_label")
            self._log_non_finite(y_true, "y_true")
        
        for epoch in range(1, self.n_epochs+1):
            
            running_loss = {mode:0.0 for mode in modes}
            
            for mode in modes:
                if mode == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                step = 0
                # This loop iterates over each batch of data for the specified mode 
                for i,(x_od_features, his_year_od_norm, sample_weight, feat_metro_geo_dist_neigh_o_d, feat_metro_link_dist_neigh_o_d, feat_metro_time_dist_neigh_o_d, \
                    timestep, feat_od_interfea_neigh_geo_dist, feat_od_interfea_neigh_link_dist, feat_od_interfea_neigh_time_dist,\
                    geo_dist_knn_metro_o_d, link_dist_knn_metro_o_d, time_dist_knn_metro_o_d, o_sta_age, d_sta_age, od_age,\
                        o_idx, d_idx, add_label, y_true) in enumerate(data_processor[mode]):
                    
                    batch_feat_neigh_od_geo = self.get_batch_feat_neigh_od(feat_metro_geo_dist_neigh_o_d, feat_od_interfea_neigh_geo_dist)
                    batch_feat_neigh_od_link = self.get_batch_feat_neigh_od(feat_metro_link_dist_neigh_o_d, feat_od_interfea_neigh_link_dist)
                    batch_feat_neigh_od_time = self.get_batch_feat_neigh_od(feat_metro_time_dist_neigh_o_d, feat_od_interfea_neigh_time_dist)
                   
                    with torch.set_grad_enabled(mode = mode=='train'):
                        outputs = self.model(x_od_features, his_year_od_norm, 
                                batch_feat_neigh_od_geo, batch_feat_neigh_od_link, batch_feat_neigh_od_time, od_age)
                        y_pred = outputs[0]
                        if self.agew:
                            losses = self.criterion(y_pred, y_true)
                            sample_weight = sample_weight.unsqueeze(1)
                            weighted_losses = losses * sample_weight
                            loss = weighted_losses.sum()
                        else:
                            loss = self.criterion(y_pred, y_true)
                        
                        if mode == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                    if self.agew:
                        running_loss[mode] += loss
                    else:
                        running_loss[mode] += loss * y_true.shape[0]
                    step += y_true.shape[0] 
                
                # compute and record the average loss per sample for each epoch
                if mode == 'train':
                    train_loss.append(running_loss[mode]/step)
                else:
                    valid_loss.append(running_loss[mode]/step)

                # epoch end
                if mode == 'valid':
                    if running_loss[mode]/step <= val_loss:
                        print(f'Epoch {epoch}, Val_loss drops from {val_loss:.5} to {running_loss[mode]/step:.5}. '
                              f'Update model checkpoint..')
                        val_loss = running_loss[mode]/step
                        checkpoint.update(epoch=epoch, state_dict=self.model.state_dict())
                        torch.save(checkpoint, model_dir + f'/{self.model_name}_best_model.pkl')
                        early_stopper = 10
                    else:
                        print(f'Epoch {epoch}, Val_loss does not improve from {val_loss:.5}.')
                        early_stopper -= 1
                        if early_stopper == 0:
                            print(f'Early stopping at epoch {epoch}..')
                            return train_loss, valid_loss
            if epoch % 20 == 0:
                self.test(epoch=epoch, data_processor=data_processor, \
                          modes=['train', 'valid', 'test'], model_dir=model_dir, \
                          data_class=self.data_class)
        # print('Training ends at: ', time.ctime())
        print('training', datetime.datetime.now() - start_time)
        
        return train_loss, valid_loss

    def test(self, epoch, data_processor:dict, modes:list, model_dir:str, data_class):

        saved_checkpoint = torch.load(model_dir + f'/{self.model_name}_best_model.pkl', map_location=torch.device('cpu'))
        self.model.load_state_dict(saved_checkpoint['state_dict'])
        self.model.eval()

        # print('Testing starts at: ', time.ctime())
        running_loss = {mode: 0.0 for mode in modes}
        start_time = datetime.datetime.now()
        if data_class.norm_opt == "minmax":
            denorm = data_class.minmax_denormalize
            params = (data_class.min1, data_class.max1)
        elif data_class.norm_opt == "std":
            denorm = data_class.std_denormalize
            params = (data_class._mean1, data_class._std1)
        else:
            raise ValueError(f"Unknown norm_opt: {data_class.norm_opt}")
        for mode in modes:
            add_ground_truth, add_prediction = list(), list()
            old_ground_truth, old_prediction = list(), list()
            ground_truth, prediction = [], []
            o_idx_ls, d_idx_ls, add_label_ls = [], [], []
            timestep_ls, od_age_ls = [], []
            weights_ls, gcn_pred_ls, temp_pred_ls = [], [], []

            for i,(x_od_features, his_year_od_norm, sample_weight, feat_metro_geo_dist_neigh_o_d, feat_metro_link_dist_neigh_o_d, feat_metro_time_dist_neigh_o_d, \
                    timestep, feat_od_interfea_neigh_geo_dist, feat_od_interfea_neigh_link_dist, feat_od_interfea_neigh_time_dist,\
                        geo_dist_knn_metro_o_d, link_dist_knn_metro_o_d, time_dist_knn_metro_o_d, o_sta_age, d_sta_age, od_age,\
                        o_idx,d_idx, add_label, y_true) in enumerate(data_processor[mode]):
                    

                timestep_ls.append(timestep.cpu().detach().numpy())
                od_age_ls.append(od_age.cpu().detach().numpy())
                o_idx_ls.append(o_idx)
                d_idx_ls.append(d_idx)
                add_label_ls.append(add_label.detach().cpu().numpy())
            
                batch_feat_neigh_od_geo = self.get_batch_feat_neigh_od(feat_metro_geo_dist_neigh_o_d, feat_od_interfea_neigh_geo_dist)
                batch_feat_neigh_od_link = self.get_batch_feat_neigh_od(feat_metro_link_dist_neigh_o_d, feat_od_interfea_neigh_link_dist)
                batch_feat_neigh_od_time = self.get_batch_feat_neigh_od(feat_metro_time_dist_neigh_o_d, feat_od_interfea_neigh_time_dist)

                y_pred, weights, gcn_pred, temp_pred = self.model(x_od_features, his_year_od_norm, 
                        batch_feat_neigh_od_geo, batch_feat_neigh_od_link, batch_feat_neigh_od_time, od_age)
                
                ground_truth.append(y_true.detach().cpu().numpy())
                prediction.append(y_pred.detach().cpu().numpy())
                weights_ls.append(weights.detach().cpu().numpy())
                gcn_pred_ls.append(gcn_pred.detach().cpu().numpy())
                temp_pred_ls.append(temp_pred.detach().cpu().numpy())
             
                old_ground_truth.append(y_true[add_label == 0].detach().cpu().numpy())
                old_prediction.append(y_pred[add_label == 0].detach().cpu().numpy())

                if torch.sum(add_label) > 0:
                    add_ground_truth.append(y_true[add_label > 0].detach().cpu().numpy())
                    add_prediction.append(y_pred[add_label > 0].detach().cpu().numpy())

            if mode == "test":
                prediction = np.concatenate(prediction, axis=0)
                ground_truth = np.concatenate(ground_truth, axis=0)
                weights_ls = np.concatenate(weights_ls, axis=0)
                gcn_pred_ls = np.concatenate(gcn_pred_ls, axis=0)
                temp_pred_ls = np.concatenate(temp_pred_ls, axis=0)
                ground_truth = denorm(ground_truth, *params)
                prediction = denorm(prediction, *params)
                o_idx_ls = np.concatenate(o_idx_ls, axis=0)
                d_idx_ls = np.concatenate(d_idx_ls, axis=0)
                timestep_ls = np.concatenate(timestep_ls, axis=0)
                add_label_ls = np.concatenate(add_label_ls, axis=0)
                od_age_ls = np.concatenate(od_age_ls, axis=0)
                save_dir = r"../results/STG_GNN"
                os.makedirs(save_dir, exist_ok=True)  
                np.save(os.path.join(save_dir, f"prediction{epoch}.npy"), prediction)
                np.save(os.path.join(save_dir, "true_all.npy"), ground_truth)
                np.save(os.path.join(save_dir, "o_idx.npy"), o_idx_ls)
                np.save(os.path.join(save_dir, "d_idx.npy"), d_idx_ls)
                np.save(os.path.join(save_dir, "year.npy"), timestep_ls)
                np.save(os.path.join(save_dir, "add_label.npy"), add_label_ls)
                np.save(os.path.join(save_dir, "od_age.npy"), od_age_ls)
                np.save(os.path.join(save_dir, "weights.npy"), weights_ls)
                np.save(os.path.join(save_dir, "gcn_pred.npy"), gcn_pred_ls)
                np.save(os.path.join(save_dir, "temp_pred.npy"), temp_pred_ls)

                all_RMSE = self.RMSE(prediction, ground_truth)
                all_MAE = self.MAE(prediction, ground_truth)
                all_CPC = self.CPC(prediction, ground_truth)
                print('All', f'{epoch}{mode} RMSE: ', all_RMSE, 'MAE:', all_MAE, 'CPC:', all_CPC)
                print("SAVE ALL PREDICT INFO...")
            
            add_RMSE, add_MAE,add_CPC = 0, 0, 0
            if len(add_ground_truth) > 0:
                add_ground_truth = np.concatenate(add_ground_truth, axis=0)
                add_prediction = np.concatenate(add_prediction, axis=0)
                add_ground_truth = denorm(add_ground_truth, *params)
                add_prediction = denorm(add_prediction, *params)
                add_RMSE = self.RMSE(add_prediction, add_ground_truth)
                add_MAE = self.MAE(add_prediction, add_ground_truth)
                add_CPC = self.CPC(add_prediction, add_ground_truth)
                print('Add', f'{epoch}{mode} RMSE: ', add_RMSE, 'MAE:', add_MAE, 'CPC:', add_CPC)
            
            old_ground_truth = np.concatenate(old_ground_truth, axis=0)
            old_prediction = np.concatenate(old_prediction, axis=0)
            old_ground_truth = denorm(old_ground_truth, *params)
            old_prediction = denorm(old_prediction, *params)
            old_RMSE = self.RMSE(old_prediction, old_ground_truth)
            old_MAE = self.MAE(old_prediction, old_ground_truth)
            old_CPC = self.CPC(old_prediction, old_ground_truth)
            print('Old', f'{epoch}{mode} RMSE: ', old_RMSE, 'MAE:', old_MAE, 'CPC:', old_CPC)
            
        print('test', datetime.datetime.now() - start_time)
        return [add_RMSE, add_MAE, add_CPC, old_RMSE, old_MAE, old_CPC]

    @staticmethod
    def RMSE(y_pred:np.array, y_true:np.array):
        return np.sqrt(np.mean(np.square(y_pred - y_true)))
    @staticmethod
    def MAE(y_pred:np.array, y_true:np.array):
        return np.mean(np.abs(y_pred - y_true))
    @staticmethod
    def CPC(y_pred: np.array, y_true: np.array):
        return (2 * np.minimum(y_pred, y_true).sum()) / (y_pred.sum() + y_true.sum())

