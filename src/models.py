# src/models.py
# Model definitions: KAN_linear, Gat_Kan_layer, KAGATFeat, CombinedModelGRU

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import numpy as np

# ------------------------ KAN Linear Layer ------------------------
class KAN_linear(nn.Module):
    def __init__(self, inputdim, outdim, gridsize, addbias=False):
        super(KAN_linear, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, outdim, inputdim, gridsize) /
            (np.sqrt(inputdim) * np.sqrt(self.gridsize))
        )
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device),
                          (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0),
                         self.fouriercoeffs)
        if self.addbias:
            y += self.bias
        y = y.view(outshape)
        return y

# ------------------------ Gat_Kan Layer ------------------------
class Gat_Kan_layer(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_node_feats,
                 out_edge_feats, num_heads, grid_size, bias=True):
        super(Gat_Kan_layer, self).__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats

        self.fc_node = nn.Linear(in_node_feats + in_edge_feats,
                                 out_node_feats * num_heads, bias=True)
        self.fc_ni = nn.Linear(in_node_feats, out_edge_feats * num_heads,
                               bias=False)
        self.fc_fij = nn.Linear(in_edge_feats, out_edge_feats * num_heads,
                                bias=False)
        self.fc_nj = nn.Linear(in_node_feats, out_edge_feats * num_heads,
                               bias=False)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads,
                                                         out_edge_feats)))
        self.output_node = KAN_linear(out_node_feats, out_node_feats,
                                      grid_size, addbias=True)
        self.output_edge = KAN_linear(out_edge_feats, out_edge_feats,
                                      grid_size, addbias=True)
        self.edge_kan = KAN_linear(out_edge_feats * num_heads,
                                   out_edge_feats * num_heads,
                                   gridsize=1, addbias=True)
        self.node_kan = KAN_linear(in_node_feats + in_edge_feats,
                                   in_node_feats + in_edge_feats,
                                   gridsize=1, addbias=True)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads *
                                                              out_edge_feats,)))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc_node.weight)
        nn.init.xavier_normal_(self.fc_ni.weight)
        nn.init.xavier_normal_(self.fc_fij.weight)
        nn.init.xavier_normal_(self.fc_nj.weight)
        nn.init.xavier_normal_(self.attn)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def message_func(self, edges):
        return {'feat': edges.data['feat']}

    def reduce_func(self, nodes):
        num_edges = nodes.mailbox['feat'].size(1)
        agg_feats = torch.sum(nodes.mailbox['feat'], dim=1) / num_edges
        return {'agg_feats': agg_feats}

    def forward(self, graph, nfeats, efeats, get_attention=False):
        with graph.local_scope():
            graph.ndata['feat'] = nfeats
            graph.edata['feat'] = efeats

            f_ni = self.fc_ni(nfeats)
            f_nj = self.fc_nj(nfeats)
            f_fij = self.fc_fij(efeats)
            graph.srcdata.update({'f_ni': f_ni})
            graph.dstdata.update({'f_nj': f_nj})
            graph.apply_edges(fn.u_add_v('f_ni', 'f_nj', 'f_tmp'))
            f_out = graph.edata.pop('f_tmp') + f_fij
            f_out = self.edge_kan(f_out)
            if self.bias is not None:
                f_out = f_out + self.bias
            f_out = F.leaky_relu(f_out)
            f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)
            e = (f_out * self.attn).sum(dim=-1).unsqueeze(-1)

            graph.send_and_recv(graph.edges(), self.message_func,
                                self.reduce_func)
            m_feats = torch.cat((graph.ndata['feat'],
                                 graph.ndata['agg_feats']), dim=1)
            m_feats = self.node_kan(m_feats)
            graph.edata['a'] = edge_softmax(graph, e)
            graph.ndata['h_out'] = self.fc_node(m_feats).view(
                -1, self._num_heads, self._out_node_feats)
            graph.update_all(fn.u_mul_e('h_out', 'a', 'm'),
                             fn.sum('m', 'h_out'))
            h_out = F.leaky_relu(graph.ndata['h_out'])
            h_out = h_out.view(-1, self._num_heads, self._out_node_feats)
            h_out = torch.sum(h_out, dim=1)
            f_out = torch.sum(f_out, dim=1)

            out_n = self.output_node(h_out)
            out_e = self.output_edge(f_out)

            if get_attention:
                return out_n, out_e, graph.edata.pop('a')
            else:
                return out_n, out_e

# ------------------------ KA-GAT Feature Extractor ------------------------
class KAGATFeat(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, hidden_dim, grid_size,
                 heads, layer_num, pooling):
        super(KAGATFeat, self).__init__()
        self.in_node_dim = in_node_dim
        self.in_edge_dim = in_edge_dim
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.heads = heads
        self.layer_num = layer_num
        self.pooling = pooling

        self.attentions = nn.ModuleList()
        self.attentions.append(
            Gat_Kan_layer(in_node_feats=in_node_dim,
                          in_edge_feats=in_edge_dim,
                          out_node_feats=hidden_dim,
                          out_edge_feats=hidden_dim,
                          num_heads=heads,
                          grid_size=grid_size)
        )
        for _ in range(layer_num - 1):
            self.attentions.append(
                Gat_Kan_layer(in_node_feats=hidden_dim,
                              in_edge_feats=hidden_dim,
                              out_node_feats=hidden_dim,
                              out_edge_feats=hidden_dim,
                              num_heads=heads,
                              grid_size=grid_size)
            )

        self.leaky_relu = nn.LeakyReLU()
        self.avgpool = dgl.nn.AvgPooling()
        self.maxpool = dgl.nn.MaxPooling()
        self.sumpool = dgl.nn.SumPooling()

    def forward(self, g, node_feat, edge_feat):
        for atten in self.attentions:
            node_feat, edge_feat = atten(g, node_feat, edge_feat)
        out_node = self.leaky_relu(node_feat)
        if self.pooling == 'avg':
            y = self.avgpool(g, out_node)
        elif self.pooling == 'max':
            y = self.maxpool(g, out_node)
        elif self.pooling == 'sum':
            y = self.sumpool(g, out_node)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return y

# ------------------------ Combined GRU Model ------------------------
class CombinedModelGRU(nn.Module):
    """
    Multi-modal model: Virus + ECFP + UniMol + KA-GAT graph features,
    followed by a GRU layer and a final linear layer.
    """
    def __init__(self, virus_dim, ecfp_dim, unimol_dim, kagat_dim,
                 gru_hidden_dim, gru_num_layers, dropout,
                 node_in_dim, edge_in_dim, hidden_dim, grid_size,
                 heads, layer_num, pooling):
        super(CombinedModelGRU, self).__init__()
        self.kagat = KAGATFeat(in_node_dim=node_in_dim,
                               in_edge_dim=edge_in_dim,
                               hidden_dim=hidden_dim,
                               grid_size=grid_size,
                               heads=heads,
                               layer_num=layer_num,
                               pooling=pooling)
        total_feat_dim = virus_dim + ecfp_dim + unimol_dim + kagat_dim
        self.gru = nn.GRU(input_size=total_feat_dim,
                          hidden_size=gru_hidden_dim,
                          num_layers=gru_num_layers,
                          batch_first=True,
                          dropout=dropout if gru_num_layers > 1 else 0)
        self.classifier = nn.Linear(gru_hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, virus, ecfp, unimol):
        kagat_feat = self.kagat(g, g.ndata['feat'], g.edata['feat'])
        x = torch.cat([virus, ecfp, unimol, kagat_feat], dim=1)
        x = x.unsqueeze(1)  # (batch, 1, total_dim)
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]   # last timestep
        gru_out = self.dropout(gru_out)
        logit = self.classifier(gru_out).squeeze(1)
        return logit