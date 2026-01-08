import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Spatial Module
class GAT(nn.Module):

    """
    Neural network block that applies attention mechanism to sampled locations (only the attention).
    """
    def __init__(self, in_channels, alpha=0.2, threshold=0.0, concat = True):
        """
        :param in_channels: Number of time step.
        :param alpha: alpha for leaky Relu.
        :param threshold: threshold for graph connection
        :param concat: whether concat features
        :It should be noted that the input layer should use linear activation
        """
        super(GAT, self).__init__()
        self.threshold = threshold
        self.concat = concat
        self.in_channels = in_channels
        self.attn1 = nn.Linear(2*in_channels, 16)
        self.attn2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, input_target, input_neigh):
        """input_target [batch_size, n_feat]"""
        """input_neigh [batch_size, n_neigh_pairs=6*6-1, n_feat]"""
        """adj [batch_size, n_neigh]"""
        """Output Shape: [batch_size, n_feat]"""

        B = input_neigh.size()[0]
        N = input_neigh.size()[1]

        a_input = torch.cat([input_target.unsqueeze(1).repeat(1, N, 1), input_neigh], -1) # batch_size, n_neigh, 2*n_feat
        e = self.leakyrelu(self.attn2(self.relu(self.attn1(a_input)))).squeeze(-1) # batch_size, n_neigh
        attention = F.softmax(e, dim=-1) # batch_size, n_neigh
        h_prime = torch.matmul(attention.unsqueeze(1), input_neigh).squeeze(1) # batch_size, 1, n_feat
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GraphConvLayer(nn.Module):
    def __init__(self, c_in, c_out, graph_conv_act_func):
        super(GraphConvLayer, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.graph_conv_act_func = graph_conv_act_func
        self.enable_bias = True
        self.align = nn.Linear(c_in, c_out) 
        self.gcnconv = GAT(c_out)

    def forward(self, x_target, x_neigh):
        x_target = x_target.to(device)
        x_neigh = x_neigh.to(device)
        x_target = self.align(x_target)
        x_neigh = self.align(x_neigh)
        x_gc_with_rc = self.gcnconv(x_target, x_neigh)
        return x_gc_with_rc


# # Temporal Module

class GRU(nn.Module):
    def __init__(self, c_out, feature_size, hidden_dim=16, nlayers=2, bidirectional=False, dropout=0.3):
        super(GRU, self).__init__()
        self.nlayers = nlayers 
        self.nhid = hidden_dim
        self.bidirectional = bidirectional
        self.encoder = nn.GRU(feature_size, hidden_dim, nlayers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.out1 = nn.Linear(nlayers * hidden_dim * (2 if bidirectional else 1), c_out)

    
    def forward(self, x_his_od):
        batch_size, _ = x_his_od.size() # batch_size, seq_len, feature_size
        x = x_his_od.unsqueeze(-1)
        h_0 = torch.zeros(self.nlayers * (2 if self.bidirectional else 1), batch_size, self.nhid, device=x.device)
        _, encoder_hidden = self.encoder(x, h_0) 
        # If bidirectional, concatenate the hidden states from both directions
        if self.bidirectional:
            encoder_hidden = encoder_hidden.view(self.nlayers, 2, batch_size, self.nhid)
            encoder_hidden = torch.cat((encoder_hidden[:, 0, :, :], encoder_hidden[:, 1, :, :]), dim=2) 
        else:
            encoder_hidden = encoder_hidden.view(self.nlayers, batch_size, self.nhid)
        # Flatten the hidden state
        t_output = encoder_hidden.transpose(0, 1).contiguous().view(batch_size, -1)  
        t_output = self.out1(t_output)
        return t_output 

class LSTM(nn.Module):
    def __init__(self, c_out, feature_size, hidden_dim=16, nlayers=2, bidirectional=False, dropout=0.3):
        super(LSTM, self).__init__()
        self.nlayers = nlayers 
        self.nhid = hidden_dim 
        self.bidirectional = bidirectional
        self.encoder = nn.LSTM(feature_size, hidden_dim, nlayers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.out1 = nn.Linear(nlayers * hidden_dim * (2 if bidirectional else 1), c_out)

    def forward(self, x_his_od):
        batch_size, _ = x_his_od.size()
        x = x_his_od.unsqueeze(-1)
        h_0 = torch.zeros(self.nlayers * (2 if self.bidirectional else 1), batch_size, self.nhid, device=x.device)
        c_0 = torch.zeros(self.nlayers * (2 if self.bidirectional else 1), batch_size, self.nhid, device=x.device)
        _, (encoder_hidden, encoder_cell) = self.encoder(x, (h_0, c_0))
        # If bidirectional, concatenate the hidden states from both directions
        if self.bidirectional:
            encoder_hidden = encoder_hidden.view(self.nlayers, 2, batch_size, self.nhid)
            encoder_hidden = torch.cat((encoder_hidden[:, 0, :, :], encoder_hidden[:, 1, :, :]), dim=2) 
        else:
            encoder_hidden = encoder_hidden.view(self.nlayers, batch_size, self.nhid)
        t_output = encoder_hidden.transpose(0, 1).contiguous().view(batch_size, -1)  # [batch_size, num_layers * hidden_dim * num_directions]
        t_output = self.out1(t_output)
        return t_output 


class CNN(nn.Module):
    def __init__(self, c_out, feature_size, hidden_dim=16, nlayers=2, dropout=0.3):
        super(CNN, self).__init__()
        self.c_out = c_out
        self.nlayers = nlayers
        self.nhid = hidden_dim
        self.conv_layers = nn.ModuleList()
        in_channels = feature_size
        
        for i in range(nlayers):
            self.conv_layers.append(
                nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
            )
            in_channels = hidden_dim
            
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(hidden_dim, c_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_his_od):
        '''x_his_od: [batch_size, seq_len]'''
        x = x_his_od.unsqueeze(-1)
        x = x.permute(0, 2, 1)  # [batch_size, feature_size, seq_len]
        for conv in self.conv_layers:
            x = self.relu(conv(x))
            x = self.dropout(x)
        x = self.global_avg_pool(x)  # [batch_size, hidden_dim, 1]
        x = x.squeeze(-1) # [batch_size, hidden_dim]
        t_output = self.fc(x)
        return t_output # [batch_size, c_out]



class STG_GNN(nn.Module):
    def __init__(
        self,
        dim_station: int,
        dim_interact: int,
        drop_rate: float = 0.5,
        n_od_gcn: int = 8,
        fc_out1: int = 256,
        fc_out2: int = 128,
        fc_out3: int = 2,
        use_transtime: bool = True,
        use_link_dist: bool = True,
        use_geo_graph: bool = True,
        temp_encoder: str = "gru",
    ):
        """
        Parameters
        ----------
        dim_station : int
            Dimension of station-level features.
        dim_interact : int
            Dimension of OD interaction features (per OD pair).
        use_transtime / use_link_dist / use_geo_graph : bool
            Whether to use each type of graph conv output in the final concat.
        temp_encoder : {"gru", "lstm", "cnn"}
            Temporal encoder type for historical OD flows.
        """
        super(STG_GNN, self).__init__()
        
        self.fc_out1 = fc_out1
        self.fc_out2 = fc_out2
        self.fc_out3 = fc_out3
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=drop_rate)
        self.use_transtime = use_transtime
        self.use_link_dist = use_link_dist
        self.use_geo_graph = use_geo_graph
        self.temp_encoder = temp_encoder

        count_true = sum([use_transtime, use_link_dist, use_geo_graph])
        assert count_true >= 1, "At least one of [transtime, link_dist, geo_graph] must be True."

        fc_in_gcn = dim_station*2 + dim_interact
        fc_in_fnn = dim_station*2 + dim_interact + count_true * n_od_gcn

        if self.temp_encoder=='gru':
            self.tmp_week = GRU(c_out=3, feature_size=1, hidden_dim=16, nlayers=1, bidirectional=False, dropout=0.0)
            self.tmp_work = GRU(c_out=3, feature_size=1, hidden_dim=16, nlayers=1, bidirectional=False, dropout=0.0)
            self.temp_fc = nn.Linear(6, fc_out3) 
        elif self.temp_encoder=='lstm':
            self.tmp_week = LSTM(c_out=3, feature_size=1, hidden_dim=16, nlayers=1, bidirectional=False, dropout=0.0)
            self.tmp_work = LSTM(c_out=3, feature_size=1, hidden_dim=16, nlayers=1, bidirectional=False, dropout=0.0)
            self.temp_fc = nn.Linear(6, fc_out3)  
        elif self.temp_encoder=='cnn':
            self.tmp_week = CNN(c_out=3, feature_size=1, hidden_dim=16, nlayers=1, dropout=0.0)
            self.tmp_work = CNN(c_out=3, feature_size=1, hidden_dim=16, nlayers=1, dropout=0.0)
            self.temp_fc = nn.Linear(6, fc_out3) 

        self.gcn_geo_dist_metro_od = GraphConvLayer(fc_in_gcn, n_od_gcn, 'relu')
        self.gcn_link_dist_metro_od = GraphConvLayer(fc_in_gcn, n_od_gcn, 'relu')
        self.gcn_time_dist_metro_od = GraphConvLayer(fc_in_gcn, n_od_gcn, 'relu')
        self.gcn_fc = nn.Sequential(
            nn.Linear(fc_in_fnn, self.fc_out1),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(self.fc_out1, self.fc_out2),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(self.fc_out2, self.fc_out3)
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(1, 16), 
            nn.ReLU(),
            nn.Linear(16, 2), 
            nn.Softmax(dim=1) 
        )

    def forward(
        self,
        x_od_features: torch.Tensor,
        his_year_od_norm: torch.Tensor,
        batch_feat_neigh_od_geo: torch.Tensor,
        batch_feat_neigh_od_link: torch.Tensor,
        batch_feat_neigh_od_time: torch.Tensor,
        od_age: torch.Tensor,
    ):
        seq_len = int(his_year_od_norm.size(1)/2)
        x_tmp_week = self.tmp_week(his_year_od_norm[:, :seq_len])
        x_tmp_work = self.tmp_work(his_year_od_norm[:, seq_len:])
        x_tmp = torch.cat([x_tmp_week, x_tmp_work], dim=-1)
        temp_pred = self.temp_fc(x_tmp)

        x_graph_geo_dist_metro_o_d = self.gcn_geo_dist_metro_od(x_od_features, batch_feat_neigh_od_geo)
        x_graph_link_dist_metro_o_d = self.gcn_link_dist_metro_od(x_od_features, batch_feat_neigh_od_link)
        x_graph_time_dist_metro_o_d = self.gcn_time_dist_metro_od(x_od_features, batch_feat_neigh_od_time)

        x_graph_list = [x_od_features]
        if self.use_transtime:
            x_graph_list.append(x_graph_time_dist_metro_o_d)
        if self.use_link_dist:
            x_graph_list.append(x_graph_link_dist_metro_o_d)
        if self.use_geo_graph:
            x_graph_list.append(x_graph_geo_dist_metro_o_d)

        x_graph = torch.cat(x_graph_list, dim=-1)

        gcn_pred = self.gcn_fc(x_graph)
        od_age = od_age.view(-1, 1) 
        weights = self.gate_net(od_age) 
        final_pred = weights[:, 0].unsqueeze(1) * gcn_pred + weights[:, 1].unsqueeze(1) * temp_pred
        return final_pred, weights, gcn_pred, temp_pred
