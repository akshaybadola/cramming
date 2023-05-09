from typing import Optional

import torch
import torch.nn as nn

from transformers import PretrainedConfig, PreTrainedModel
from omegaconf import OmegaConf

from .components import (
    INPLACE,
    _get_layer_fn,
    _get_norm_fn,
    EmbeddingComponent,
    PoolingComponent,
    PredictionHeadComponent,
    Sequential,
    get_extended_attention_mask,
)
from .scriptable_bert import _get_loss_fn, _init_module, crammedBertConfig
from .parallel_masked_lm import ScriptableLMForPreTraining


def construct_tree_masked_lm(cfg_arch, vocab_size, downstream_classes=None):
    cfg_arch.embedding.vocab_size = vocab_size
    cfg_arch.num_labels = downstream_classes
    config = crammedBertConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    if downstream_classes is None:
        model = ScriptableLMForPreTraining(TreeLM(cfg_arch), cfg_arch)
    else:
        model = ScriptableLMForSequenceClassification(config)
    return model


class TriBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()
        self.block_a = FlatBlock(dim, heads, dim_head, mlp_dim)
        self.block_b = FlatBlock(dim, heads, dim_head, mlp_dim)
        self.w = nn.Linear(dim, dim)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.w(self.silu(self.block_a(x)) * self.block_b(x))


class IDBlock(nn.Module):
    def __init__(self, idx, cfg_arch):
        super().__init__()
        self.block = _get_layer_fn(cfg_arch.layer_macro_type)(idx, cfg_arch)
        self.w = nn.Linear(cfg_arch.hidden_size, cfg_arch.hidden_size)
        self.silu = nn.SiLU()

    def forward(self, states, attention_mask, res_scale=1):
        states = self.block(states, attention_mask, res_scale)
        return self.w(self.silu(states) * states)


class DownSample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, states, attention_mask, res_scale=None):
        shape = states.shape
        if shape[0] // 2:
            states = states.reshape([shape[0]//2, 2, shape[1], shape[2], shape[3]])
            return states.mean(1)
        else:
            return states


class TreeLayer(nn.Module):
    def __init__(self, idx, cfg_arch, width):
        super().__init__()
        self.width = width
        self.blocks = nn.ModuleList([IDBlock(idx, cfg_arch) for _ in range(width)])

    def forward(self, states, attention_mask, res_scale):
        output = torch.zeros_like(states)
        for i, block in enumerate(self.blocks):
            output[i] = block(states[i], attention_mask, res_scale)
        return output


class TreeLM(torch.nn.Module):
    def __init__(self, cfg_arch):
        super().__init__()
        self.cfg = cfg_arch

        self.embedding = EmbeddingComponent(cfg_arch.embedding, cfg_arch.norm, cfg_arch.norm_eps)
        if cfg_arch.embedding.embedding_dim == cfg_arch.hidden_size:
            self.input_projection = torch.nn.Identity()
        else:
            self.input_projection = torch.nn.Linear(
                cfg_arch.embedding.embedding_dim,
                cfg_arch.hidden_size,
                bias=cfg_arch.use_bias,
            )
        if cfg_arch.recurrent_layers is not None:
            raise NotImplementedError("Recurrent layers not implemented in Tree like LM")
        w = cfg_arch.num_transformer_layers
        idx = 0
        layers = []
        while w:
            layers.append(TreeLayer(idx, cfg_arch, w))
            layers.append(DownSample())
            w //= 2
            idx += 1
        # NOTE: Fusion is tricky here because of some partial functions deeper down
        # if self.cfg.layer_fusion:
        #     self.layers = torch.jit.script(nn.Sequential(*self.layers))
        # else:
        #     self.layers = nn.Sequential(*self.layers)
        self.layers = nn.Sequential(*layers)
        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()
        self.num_layers = self.cfg.num_transformer_layers

        self.seq_first = self.layers[0].blocks[0].block.LAYOUT == "[S B H]" if len(self.layers) > 0 else False
        self.gradient_checkpointing = cfg_arch.gradient_checkpointing
        if self.gradient_checkpointing:
            raise NotImplementedError("Gradient Checkpointing not implemented")
        self.layer_drop_theta = cfg_arch.layer_drop_theta
        if self.layer_drop_theta:
            raise NotImplementedError("Layer Drop not implemented")
        self.register_buffer("p", torch.tensor(1.0))  # Layer scaling factor # Assign this only once

    def _init_weights(self, *args, **kwargs):
        for name, module in self.named_modules():
            _init_module(
                name,
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape,
                                                         self.cfg.attention.causal_attention)
        hidden_states = self.input_projection(self.embedding(input_ids))
        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        hidden_states = hidden_states.unsqueeze(0).expand(self.num_layers, *hidden_states.shape)
        # Main transformer blocks:
        if self.gradient_checkpointing and self.training:
            # Hide this away from any jit-ing...
            hidden_states = self.forward_checkpointed(hidden_states, attention_mask)
        else:
            if self.layer_drop_theta is None:
                for i, layer_module in enumerate(self.layers):
                    hidden_states = layer_module(hidden_states, attention_mask, self.p)
            else:
                p = self.p.clone()
                step = (1 - self.layer_drop_theta) / len(self.layers)
                for i, layer_module in enumerate(self.layers):
                    p = p - step
                    if torch.bernoulli(p):
                        hidden_states = layer_module(hidden_states, attention_mask, res_scale=1 / p)
        hidden_states = hidden_states.squeeze(0)
        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        return self.final_norm(hidden_states)

    @torch.jit.ignore
    def forward_checkpointed(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        if self.layer_drop_theta is None:
            for i, layer_module in enumerate(self.layers):
                hidden_states = torch.utils.checkpoint.checkpoint(layer_module, hidden_states, attention_mask)
        else:
            p = self.p.clone()
            step = (1 - self.layer_drop_theta) / len(self.layers)
            for i, layer_module in enumerate(self.layers):
                p = p - step
                if torch.bernoulli(p):
                    hidden_states = torch.utils.checkpoint.checkpoint(layer_module, hidden_states, attention_mask, res_scale=1 / p)
        return hidden_states
            
