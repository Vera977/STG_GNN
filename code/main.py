"""This code is partly adapted from: https://github.com/yuebingliang/ZINB-GNN-for-sparse-OD-flow-generation"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from data import DataInput, DataGenerator
from models import STG_GNN
from trainer import ModelTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transtime", action="store_true",
                        help="Use transfer-aware travel-time graph as one GNN input.")
    parser.add_argument("--link_dist", action="store_true",
                        help="Use graph link-distance graph as one GNN input.")
    parser.add_argument("--geo_graph", action="store_true",
                        help="Use geographical-distance graph as one GNN input.")
    parser.add_argument("--agew", action="store_true",
                        help="Enable age-based sample weighting.")
    parser.add_argument('--temp_encoder', type=str, default='gru',
                        choices=['gru', 'lstm', 'cnn'],
                        help='Temporal encoder type.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')

    return parser

def setup_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = build_parser()
    args = parser.parse_args()
    print('Parsed args:', args)
    run_model(args)

def run_model(args):

    year_metro_od_weekend_global = '../synthetic_data/year_od_weekend_syn_19.npy'
    year_metro_od_workday_global = '../synthetic_data/year_od_workday_syn_19.npy'
    data_dir={"weekend": year_metro_od_weekend_global, "workday": year_metro_od_workday_global}
    geo_dist_knn_metro = '../synthetic_data/geo_dist_knn_metro.npy' 
    link_dist_knn_metro = '../synthetic_data/link_dist_knn_metro.npy' 
    func_knn_metro = '../synthetic_data/func_knn_metro.npy'
    trans_time_knn_metro = '../synthetic_data/trans_time_knn_metro.npy'
    knn_dir={"metro": {"geo_dist": geo_dist_knn_metro,"link_dist": link_dist_knn_metro, 
                    "func": func_knn_metro, "transtime":trans_time_knn_metro}}
    sta_month_dir = '../synthetic_data'
    exist_od_mask_dir = '../synthetic_data/exist_od_mask.npy'
    exist_sta_mask_dir = '../synthetic_data/exist_sta_mask.npy'
    exist_mask_dir = {"od": exist_od_mask_dir, "sta": exist_sta_mask_dir}
    sta_feat_path = '../synthetic_data/year_sta_features.npy'
    od_interfea_path = '../synthetic_data/od_interfea.npy'

    data_input= DataInput(data_dir=data_dir, knn_dir=knn_dir, exist_mask_dir=exist_mask_dir,\
                            sta_month_dir=sta_month_dir, sta_feat_path=sta_feat_path, \
                            od_interfea_path=od_interfea_path, k1=5, norm_opt="std")
    data = data_input.load_data()

    batch_size = 32
    data_generator = DataGenerator()
    data_processor = data_generator.get_data_loader(data=data, batch_size=batch_size, device=device, data_class=data_input)

    """Train model (GAT)"""
    add_RMSE_ls, add_MAE_ls, add_CPC_ls = [], [], []
    old_RMSE_ls, old_MAE_ls, old_CPC_ls = [], [], []
    total_train_loss_ls, total_valid_loss_ls = [], []
    for ep in range(5, 6):
        epoch=21
        learn_rate, weight_decay = args.lr, 1e-5
        if args.agew:
            loss = nn.MSELoss(reduction='none') # for weighted loss
        else:
            loss = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam
        model_name = "STG_GNN"
        print(model_name)
        seed = ep
        setup_seed(seed)
        model = STG_GNN(
            dim_station=data_input.n_sta_feat,
            dim_interact=4,
            use_transtime=args.transtime,
            use_link_dist=args.link_dist,
            use_geo_graph=args.geo_graph,
            temp_encoder=args.temp_encoder,  
        ).to(device)
        trainer = ModelTrainer(model_name=model_name, model=model, loss=loss, optimizer=optimizer, lr=learn_rate, \
                            wd=weight_decay, n_epochs=epoch, data_class=data_input, agew=args.agew)
        model_dir = r"../results/STG_GNN/model"
        os.makedirs(model_dir, exist_ok=True)
        train_loss_ls, valid_loss_ls = trainer.train(data_processor=data_processor, modes=["train", "valid"], model_dir=model_dir, data_class=data_input)
        total_train_loss_ls.append([i.item() for i in train_loss_ls])
        total_valid_loss_ls.append([i.item() for i in valid_loss_ls])
        res_ls = trainer.test(epoch=ep, data_processor=data_processor, modes=["train", "test"], model_dir=model_dir, data_class=data_input)
        add_RMSE_ls.append(res_ls[0])
        add_MAE_ls.append(res_ls[1])
        add_CPC_ls.append(res_ls[2])
        old_RMSE_ls.append(res_ls[3])
        old_MAE_ls.append(res_ls[4])
        old_CPC_ls.append(res_ls[5])

    add_RMSE = np.mean(np.array(add_RMSE_ls))
    add_MAE = np.mean(np.array(add_MAE_ls))
    add_CPC = np.mean(np.array(add_CPC_ls))
    old_RMSE = np.mean(np.array(old_RMSE_ls))
    old_MAE = np.mean(np.array(old_MAE_ls))
    old_CPC = np.mean(np.array(old_CPC_ls))

    print('Mean Add', 'RMSE: ', add_RMSE, 'MAE:', add_MAE, 'CPC:', add_CPC)
    print('Mean Old', 'RMSE: ', old_RMSE, 'MAE:', old_MAE, 'CPC:', old_CPC)

    """save results"""
    df = pd.DataFrame(data={"add_RMSE": add_RMSE_ls, "add_MAE": add_MAE_ls, "add_CPC": add_CPC_ls,
                        "old_RMSE": old_RMSE_ls, "old_MAE": old_MAE_ls, "old_CPC": old_CPC_ls})
    save_dir = r"../results/STG_GNN"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name + ".csv")
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()