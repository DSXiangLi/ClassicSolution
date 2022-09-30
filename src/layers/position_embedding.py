# -*-coding:utf-8 -*-
from torch import nn
import torch


class SinusoidalPE(nn.Module):
    def __init__(self, output_dim):
        super(SinusoidalPE, self).__init__()
        self.output_dim = output_dim

    def forward(self, inputs):
        """
        Inputs: batch_size * seq_Len * emb_dim
        Output: batch_size * seq_len * output_dim
        """
        batch_size, seq_len, *args = inputs.shape
        position_ids = torch.arange(seq_len, dtype=torch.float).unsqueeze(-1)  # seq_len * 1
        wk = torch.pow(10000.0, -2 * torch.arange(self.output_dim // 2) / self.output_dim)
        pe = position_ids * wk
        pe = torch.stack([torch.sin(pe), torch.cos(pe)], dim=-1)  # seq_len * output_dim/2 * 2
        pe = pe.repeat((batch_size, *([1] * len(pe.shape))))  # batch_size * seq_len * output_dim/2 * 2
        pe = torch.reshape(pe, [batch_size, seq_len, self.output_dim])  # reshape into [cos, sin, cos, sin]
        return pe.to(inputs.device)


class RoPE(SinusoidalPE):
    def __init__(self, output_dim):
        super(RoPE, self).__init__(output_dim)

    def forward(self, inputs):
        """
        Rotary Position Embedding
            [p0, p1, p2,p3] * [cos(pos*w0), cos(pos*w0), cos(pos*w1), cos(pos*w1)] +
            [-p1,p0,-p3,p2] * [sin(pos*w0), sin(pos*w0), sin(pos*w1), sin(pos*w1)]
        Input: embedding [batch_size * seq_len * (*dims) * emb_dim] 可以为3维或者4维
        Output: embedding with rotary position encoding, same shape as input
        """
        # inputs:
        pe = super().forward(inputs)  # batch_size * seq_len * output_dim
        cos_pos = pe[..., 1::2].repeat_interleave(repeats=2, dim=-1)  # [cos(pos*w0), cos(pos*w0)]
        sin_pos = pe[..., 1::2].repeat_interleave(repeats=2, dim=-1)  # [sin(pos*w0), sin(pos*w0)]
        if len(inputs.shape) == 4:
            #  如果输入为4维默认在倒数第二个维度存在额外的dimension
            cos_pos = cos_pos.unsqueeze(-2)
            sin_pos = sin_pos.unsqueeze(-2)
        inputs2 = torch.stack([-inputs[..., 1::2], inputs[..., ::2]], dim=-1)  # [[-p1,-p3,-p5],[p0,p2,p4]]
        inputs2 = inputs2.reshape(inputs.shape)  # [-p1, p0, -p3, p2]
        output = inputs * cos_pos + inputs2 * sin_pos
        return output


if __name__ == '__main__':
    spe = SinusoidalPE(4)
    inputs = torch.tensor([[[1, 2, 3, 4], [11, 22, 33, 44]]])
    print(inputs.shape)
    pe = spe(inputs)
    print(pe.shape, pe)

    rpe = RoPE(4)
    inputs = torch.tensor([[[1, 2, 3, 4], [11, 22, 33, 44]]])
    print(inputs.shape)
    output = rpe(inputs)
    print(output.shape, output)
