"""Runtime patches for inference-time ablation."""
import os

_orig_tsa_forward = None
_orig_oc_forward = None


def _patch_tsa_forward():
    from OpenCity.OpenCity import TemporalSelfAttention

    global _orig_tsa_forward
    if _orig_tsa_forward is None:
        _orig_tsa_forward = TemporalSelfAttention.forward

    def forward(self, x_q, x_k, x_v, TH, TP, adj, geo_mask=None, sem_mask=None, trg_mask=False):
        mode = os.environ.get("OPENCITY_ABLATION", "full")
        if mode == "full":
            return _orig_tsa_forward(self, x_q, x_k, x_v, TH, TP, adj, geo_mask, sem_mask, trg_mask)

        B, T_q, N, D = x_q.shape
        T_k, T_v = x_k.shape[1], x_v.shape[1]

        if mode == "minus_PTTM":
            tc_x = x_q
        else:
            tc_q = self.tc_q_conv(TP).transpose(1, 2)
            tc_k = self.tc_k_conv(TH).transpose(1, 2)
            tc_v = self.tc_v_conv(x_q).transpose(1, 2)
            tc_q = tc_q.reshape(B, N, T_q, self.tc_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            tc_k = tc_k.reshape(B, N, T_k, self.tc_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            tc_v = tc_v.reshape(B, N, T_v, self.tc_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            tc_attn = (tc_q @ tc_k.transpose(-2, -1)) * self.scale
            tc_attn = tc_attn.softmax(dim=-1)
            tc_attn = self.tc_attn_drop(tc_attn)
            tc_x = (tc_attn @ tc_v).transpose(2, 3).reshape(B, N, T_q, D).transpose(1, 2)
            tc_x = self.norm_tatt1(tc_x + x_q)

        if mode == "minus_DTP":
            t_x = tc_x
        else:
            t_q = self.t_q_conv(tc_x).transpose(1, 2)
            t_k = self.t_k_conv(tc_x).transpose(1, 2)
            t_v = self.t_v_conv(tc_x).transpose(1, 2)
            t_q = t_q.reshape(B, N, T_q, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            t_k = t_k.reshape(B, N, T_k, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            t_v = t_v.reshape(B, N, T_v, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
            t_attn = t_attn.softmax(dim=-1)
            t_attn = self.t_attn_drop(t_attn)
            t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T_q, D).transpose(1, 2)
            t_x = self.norm_tatt2(t_x + tc_x)

        if mode == "minus_SDM":
            x = self.proj_drop(t_x)
        else:
            gcn_out = self.GCN(t_x, adj)
            x = self.proj_drop(gcn_out)
        return x

    TemporalSelfAttention.forward = forward


def _patch_opencity_forward():
    from OpenCity.OpenCity import OpenCity

    global _orig_oc_forward
    if _orig_oc_forward is None:
        _orig_oc_forward = OpenCity.forward

    def forward(self, input, lbls, select_dataset):
        mode = os.environ.get("OPENCITY_ABLATION", "full")
        if mode != "minus_STC":
            return _orig_oc_forward(self, input, lbls, select_dataset)

        import torch
        bs, time_steps, num_nodes, num_feas = input.size()
        TCH = input[..., self.output_dim:].long()
        TCP = lbls[..., self.output_dim:].long()
        zero_t = torch.zeros_like(TCH)
        zero_p = torch.zeros_like(TCP)
        feas_all_his, feas_all_pre = self.patch_embedding_time(torch.cat([zero_t, zero_p], dim=-1))
        spa_feas = self.spatial_embedding(self.lap_mx_dict[select_dataset].to(self.device) * 0)
        spa_feas = spa_feas.repeat(bs, feas_all_his.shape[1], 1, 1)
        feas_all_his = feas_all_his + spa_feas
        feas_all_pre = feas_all_pre + spa_feas

        x_in = input[..., : self.output_dim]
        means = x_in.mean(1, keepdim=True).detach()
        x_in = x_in - means
        stdev = torch.sqrt(torch.var(x_in, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_in /= stdev
        enc = self.patch_embedding_flow(x_in)
        adj = self.adj_mx_dict[select_dataset].to(self.device)
        for encoder_block in self.encoder_blocks:
            enc = encoder_block(
                enc, enc, enc, feas_all_his, feas_all_pre, adj,
                self.geo_mask_dict[select_dataset].to(self.device), self.sem_mask,
            )
        skip = enc.permute(0, 2, 3, 1).contiguous()
        skip = self.flatten(skip)
        skip = self.linear(skip).transpose(1, 2).unsqueeze(-1)
        skip = skip[:, :time_steps, :, :]
        skip = skip * stdev + means
        return skip

    OpenCity.forward = forward


def apply(ablation: str = "full"):
    os.environ["OPENCITY_ABLATION"] = ablation
    _patch_tsa_forward()
    _patch_opencity_forward()
