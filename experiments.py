import hydra
import torch

import cramming
from cramming import utils


hydra.initialize(config_path="cramming/config")


def get_model(arch, checkpoint_path):
    cfg = hydra.compose(config_name="cfg_eval", overrides=["arch=bert-c5"])
    cfg.eval.checkpoint = checkpoint_path
    tokenizer, cfg_arch, model_file = utils.find_pretrained_checkpoint(cfg)
    model = cramming.construct_model(cfg_arch, tokenizer.vocab_size)
    state_dict = torch.load(model_file, map_location="cpu")
    if isinstance(state_dict, list):
        model.load_state_dict(state[1])
    elif isinstance(state_dict, dict):
        model.load_state_dict(state_dict["model_state"])
    return model, tokenizer
