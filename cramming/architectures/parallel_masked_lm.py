from typing import Optional

import torch
import torch.nn as nn

from transformers import PretrainedConfig, PreTrainedModel
from omegaconf import OmegaConf

from .components import (
    _get_layer_fn,
    _get_norm_fn,
    EmbeddingComponent,
    PoolingComponent,
    PredictionHeadComponent,
    Sequential,
    get_extended_attention_mask,
)
from .scriptable_bert import _get_loss_fn, _init_module, crammedBertConfig

def construct_parallel_masked_lm(cfg_arch, vocab_size, downstream_classes=None):
    cfg_arch.embedding.vocab_size = vocab_size
    cfg_arch.num_labels = downstream_classes
    config = crammedBertConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    if downstream_classes is None:
        model = ScriptableLMForPreTraining(ParallelMaskedLM(cfg_arch), cfg_arch)
    else:
        model = ScriptableLMForSequenceClassification(config)
    return model

class ParallelMaskedLM(torch.nn.Module):
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
        layer_fn = _get_layer_fn(cfg_arch.layer_macro_type)
        if cfg_arch.recurrent_layers is None:
            self.layers = torch.nn.ModuleList([layer_fn(idx, cfg_arch)
                                               for idx in range(cfg_arch.num_transformer_layers)])
        else:
            core_block = Sequential([layer_fn(idx, cfg_arch)
                                     for idx in range(cfg_arch.recurrent_layers)])
            self.layers = torch.nn.ModuleList([core_block
                                               for _ in range(cfg_arch.num_transformer_layers)])
        self.pre_fc = layer_fn(0, cfg_arch)
        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()
        self.num_layers = self.cfg.num_transformer_layers

        self.seq_first = self.layers[0].LAYOUT == "[S B H]" if len(self.layers) > 0 else False
        self.gradient_checkpointing = cfg_arch.gradient_checkpointing
        if self.gradient_checkpointing:
            raise NotImplementedError("Gradient Checkpointing not implemented")
        self.layer_drop_theta = cfg_arch.layer_drop_theta
        if self.layer_drop_theta:
            raise NotImplementedError("Layer Drop not implemented")
        self.register_buffer("p", torch.tensor(1.0))  # Layer scaling factor # Assign this only once


    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask,
                                                         input_ids.shape,
                                                         self.cfg.attention.causal_attention)
        hidden_states = self.input_projection(self.embedding(input_ids))

        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        hidden_states = hidden_states.unsqueeze(0).expand(self.num_layers, *hidden_states.shape)
        outputs = torch.zeros_like(hidden_states)
        for i, layer_module in enumerate(self.layers):
            outputs[i] = layer_module(hidden_states[i], attention_mask, self.p)
        outputs = outputs.sum(0)

        outputs = self.pre_fc(outputs, attention_mask, self.p)
        if self.seq_first:
            outputs = outputs.transpose(0, 1).contiguous()

        return self.final_norm(outputs)

    @torch.jit.ignore
    def forward_checkpointed(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        if self.layer_drop_theta is None:
            for i, layer_module in enumerate(self.layers):
                hidden_states = torch.utils.checkpoint.checkpoint(layer_module,
                                                                  hidden_states,
                                                                  attention_mask)
        else:
            p = self.p.clone()
            step = (1 - self.layer_drop_theta) / len(self.layers)
            for i, layer_module in enumerate(self.layers):
                p = p - step
                if torch.bernoulli(p):
                    hidden_states = torch.utils.checkpoint.checkpoint(layer_module,
                                                                      hidden_states,
                                                                      attention_mask,
                                                                      res_scale=1/p)
        return hidden_states


class ScriptableLMForPreTraining(torch.nn.Module):
    """Definitely can represent BERT, but also a lot of other things. To be used for MLM schemes."""

    config_class = crammedBertConfig

    def __init__(self, encoder, cfg_arch):
        super().__init__()
        self.cfg = cfg_arch

        self.encoder = encoder
        if not cfg_arch.skip_head_transform:
            self.prediction_head = PredictionHeadComponent(cfg_arch)
        else:
            self.prediction_head = torch.nn.Linear(
                cfg_arch.hidden_size,
                cfg_arch.embedding.embedding_dim,
                bias=cfg_arch.use_bias,
            )

        if cfg_arch.loss == "szegedy":
            self.decoder = torch.nn.Identity()
        else:
            if self.cfg.tie_weights:
                self.decoder = torch.nn.Linear(cfg_arch.embedding.embedding_dim,
                                               cfg_arch.embedding.vocab_size,
                                               bias=cfg_arch.decoder_bias)
                self.decoder.weight = self.encoder.embedding.word_embedding.weight
            else:
                self.decoder = torch.nn.Linear(cfg_arch.hidden_size,
                                               cfg_arch.embedding.vocab_size,
                                               bias=cfg_arch.decoder_bias)

        self.loss_fn = _get_loss_fn(cfg_arch.loss,
                                    z_loss_factor=cfg_arch.z_loss_factor,
                                    embedding=self.encoder.embedding.word_embedding)
        self.sparse_prediction = self.cfg.sparse_prediction
        self.vocab_size = cfg_arch.embedding.vocab_size

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
        outputs = self.encoder(input_ids, attention_mask)
        outputs = outputs.view(-1, outputs.shape[-1])

        if self.sparse_prediction:
            masked_lm_loss = self._forward_dynamic(outputs, labels)
        else:
            outputs = self.decoder(self.prediction_head(outputs))
            if labels is not None:
                masked_lm_loss = self.loss_fn(outputs, labels.view(-1))
            else:
                masked_lm_loss = outputs.new_zeros((1,))

        return dict(loss=masked_lm_loss)

    # Sparse prediction can have an unpredictable number of entries in each batch
    # depending on how MLM is running
    # for this reason, the code has to fall back to eager mode there
    # @torchdynamo.disable
    def _forward_dynamic(self, outputs: torch.Tensor, labels: Optional[torch.Tensor] = None):
        if labels is not None:
            labels = labels.view(-1)
            mask_positions = labels.view(-1) != self.loss_fn.ignore_index
            outputs = outputs[mask_positions]
            labels = labels[mask_positions]

        outputs = self.decoder(self.prediction_head(outputs))
        if labels is not None:
            masked_lm_loss = self.loss_fn(outputs, labels)
        else:
            masked_lm_loss = outputs.new_zeros((1,))
        return masked_lm_loss


class ScriptableLMForSequenceClassification(PreTrainedModel):
    """Classification head and pooler."""

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)  # this could be nicer ...
        self.encoder = ParallelMaskedLM(self.cfg)

        self.pooler = PoolingComponent(self.cfg.classification_head, self.cfg.hidden_size)
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.cfg.num_labels)

        self.problem_type = None
        self.num_labels = self.cfg.num_labels
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            _init_module(
                name,
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        logits = self.head(self.pooler(self.encoder(input_ids, attention_mask)))

        if labels is not None:
            if self.problem_type is None:  # very much from huggingface
                if self.cfg.num_labels == 1:
                    self.problem_type = "regression"
                elif self.cfg.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        else:
            loss = logits.new_zeros((1,))

        return dict(logits=logits, loss=loss)
