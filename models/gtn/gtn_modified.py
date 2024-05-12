# @Time    : 2021/07/21 19:28
# @Author  : SY.M
# @FileName: transformer.py

# SOURCE: https://github.com/ZZUFaceBookDL/Gated_Transformer_Network/blob/master/Gated_Transfomer_Network/module/for_MTS/transformer.py

import torch

from encoder import Encoder
from embedding import Embedding


class GTN_Modified(torch.nn.Module):
    def __init__(self, config):
        super(GTN_Modified, self).__init__()

        d_model = config.d_model
        d_hidden = config.d_hidden
        d_feature = config.d_feature
        d_timestep = config.d_timestep
        q = config.q
        v = config.v
        h = config.h
        N = config.N
        head_hidden = config.head_hidden
        class_num = config.class_num
        dropout = config.dropout

        self.timestep_embedding = Embedding(d_feature=d_feature, d_timestep=d_timestep, d_model=d_model, wise='timestep')
        self.feature_embedding = Embedding(d_feature=d_feature, d_timestep=d_timestep, d_model=d_model, wise='feature')

        self.timestep_encoderlist = torch.nn.ModuleList([Encoder(
            d_model=d_model,
            d_hidden=d_hidden,
            q=q,
            v=v,
            h=h,
            dropout=dropout) for _ in range(N)])

        self.feature_encoderlist = torch.nn.ModuleList([Encoder(
            d_model=d_model,
            d_hidden=d_hidden,
            q=q,
            v=v,
            h=h,
            dropout=dropout) for _ in range(N)])
        
        self.head_relu = torch.nn.ReLU()
        self.head_dropout = torch.nn.Dropout(p=dropout)
        self.head_bn = torch.nn.Dropout(p=dropout)
        
        self.timestep_avg_pool = torch.nn.AvgPool1d(d_timestep)
        self.feature_avg_pool = torch.nn.AvgPool1d(d_feature)

        self.gate = torch.nn.Sequential(
            torch.nn.Linear(in_features=d_model*2, out_features=d_model),
            self.head_relu,
            torch.nn.BatchNorm1d(d_model),
            self.head_dropout,
            torch.nn.Linear(in_features=d_model, out_features=head_hidden),
            self.head_relu,
            torch.nn.BatchNorm1d(head_hidden),
            self.head_dropout,
            torch.nn.Linear(in_features=head_hidden, out_features=2)
        )

        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=d_model, out_features=head_hidden),
            self.head_relu,
            torch.nn.BatchNorm1d(head_hidden),
            self.head_dropout,
            torch.nn.Linear(in_features=head_hidden, out_features=class_num)
        )

    def forward(self, x: torch.Tensor):
        
        x = x.permute(0, 2, 1)

        x_timestep, _ = self.timestep_embedding(x)
        x_feature, _ = self.feature_embedding(x)

        for encoder in self.timestep_encoderlist:
            x_timestep, heatmap = encoder(x_timestep)

        for encoder in self.feature_encoderlist:
            x_feature, heatmap = encoder(x_feature)

        # print("x_timestep", x_timestep.shape)
        # print("x_feature", x_feature.shape)

        x_timestep = x_timestep.permute(0, 2, 1)
        x_feature = x_feature.permute(0, 2, 1)

        # print("x_timestep", x_timestep.shape)
        # print("x_feature", x_feature.shape)

        x_timestep_pooled = self.timestep_avg_pool(x_timestep).squeeze(2)
        x_feature_pooled = self.feature_avg_pool(x_feature).squeeze(2)

        # print("x_timestep_pooled", x_timestep_pooled.shape)
        # print("x_feature_pooled", x_feature_pooled.shape)

        x_concat = torch.cat([x_timestep_pooled, x_feature_pooled], dim=-1)

        # print("x_concat", x_concat.shape)


        gate_weights = torch.nn.functional.softmax(self.gate(x_concat), dim=-1)

        # print("gate_weights", gate_weights.shape)

        gate_out = x_timestep_pooled * gate_weights[:, 0:1] + x_feature_pooled * gate_weights[:, 1:2]

        out = self.cls_head(gate_out)

        return out