import hydra
import torch

import cramming
from cramming import utils


hydra.initialize(config_path="cramming/config")


def load_fixing_names(model, state_dict):
    model_keys = [*model.state_dict().keys()]
    keys = [*state_dict.keys()]
    for k in keys:
        if k in model_keys:
            continue
        if k.replace("module.", "") in model_keys:
            state_dict[k.replace("module.", "")] = state_dict.pop(k)
    return model.load_state_dict(state_dict, strict=False)


def get_model(arch, checkpoint_path, tokenizer_path):
    cfg = hydra.compose(config_name="cfg_eval", overrides=[f"arch={arch}"])
    cfg.eval.checkpoint = checkpoint_path
    tokenizer, cfg_arch, model_file = utils.find_pretrained_checkpoint(cfg, tokenizer_path=tokenizer_path)
    model = cramming.construct_model(cfg_arch, tokenizer.vocab_size)
    state = torch.load(model_file, map_location="cpu")
    if isinstance(state, list):
        load_fixing_names(model, state[1])
    elif isinstance(state, dict):
        load_fixing_names(model, state["model_state"])
    return model, tokenizer
