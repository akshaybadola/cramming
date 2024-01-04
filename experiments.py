import hydra
import torch
from functools import partial

import cramming
from cramming import utils

import matplotlib.pyplot as plt
import seaborn as sb

plt.rcParams['savefig.facecolor']='white'


hydra.initialize(config_path="cramming/config")


class CollectOutputs:
    def __init__(self, model):
        self.model = model
        self.model = self.model.eval()
        self._module_names = [x[0] for x in model.named_modules()]
        self._results = {}
        self._handles = {}

    def clear_all_hooks(self):
        for v in self._handles.values():
            v.remove()
        self._handles = {}

    def remove_hooks(self, module_names=None):
        if not module_names:
            module_names = self._module_names
        for k in module_names:
            v = self._handles.pop(k)
            v.remove()

    def debug_hook(self, module_name, *x):
        import ipdb; ipdb.set_trace()

    def add_debug_hook(self, module_names):
        if self._handles:
            for v in self._handles.values():
                v.remove()
            self._handles = {}
        self._handles = {module_name: self.model.get_submodule(module_name).
                         register_forward_hook(partial(self.debug_hook, module_name))
                         for module_name in module_names}

    def collect_hook(self, module_name, *x):
        if module_name not in self._results:
            self._results[module_name] = []
        self._results[module_name].append((x[0], x[1], x[2]))

    def add_collect_hook(self, module_names):
        if self._handles:
            for v in self._handles.values():
                v.remove()
            self._handles = {}
        self._handles = {module_name: self.model.get_submodule(module_name).
                         register_forward_hook(partial(self.collect_hook, module_name))
                         for module_name in module_names}

    def get_result_for(self, module_name, indx=-1):
        mod, inputs, outputs = self._results[module_name][indx]
        return {"module": mod, "input": inputs, "output": outputs}



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
    cfg.impl.no_jit_compilation = True
    cfg.impl.jit_instruction_type = None
    cfg.arch.layer_fusion = False
    cfg.arch.attention.high_level_fusion = False
    cfg.arch.attention.low_level_fusion = False
    tokenizer, cfg_arch, model_file = utils.find_pretrained_checkpoint(cfg, tokenizer_path=tokenizer_path)
    model = cramming.construct_model(cfg_arch, tokenizer.vocab_size)
    state = torch.load(model_file, map_location="cpu")
    if isinstance(state, list):
        load_fixing_names(model, state[1])
    elif isinstance(state, dict):
        load_fixing_names(model, state["model_state"])
    return model, tokenizer


def demo(arch, checkpoint_path, tokenizer_path):
    model, tokenizer = get_model(arch, checkpoint_path, tokenizer_path)
    collect = CollectOutputs(model)
    collect.add_collect_hook(['encoder.layers.0.attn'])
    text = "There is a light that"
    encoded_input = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    print(collect.get_result_for('encoder.layers.0.attn'))


def collect_example():
    model, tokenizer = get_model("bert-with-norm-output-tiny",
                                 "/home/joydipb01/Documents/sem9/Thesis/norm-analysis-of-transformer-20231026T045237Z-001/norm-analysis-of-transformer/cramming/2.9051.pth",
                                 "/home/joydipb01/Documents/sem9/Thesis/norm-analysis-of-transformer-20231026T045237Z-001/norm-analysis-of-transformer/cramming/tokenizer/")

    #model, tokenizer = test_bert_with_norm_output()
    model = model.eval()
    collect = CollectOutputs(model)
    collect.add_collect_hook(['encoder.layers.0.attn'])
    text = "There is a light that"
    encoded_input = tokenizer(text, return_tensors='pt')
    tokens=tokenizer.convert_ids_to_tokens(encoded_input['input_ids'].tolist()[0])
    with torch.no_grad():
        output = model(**encoded_input)
    return model._results, tokens              # this will have attention scores

def plot_weights():
    results, toks=collect_example()
    new_results={key: [item['probs'] for item in value] for key, value in results.items()}

    print(toks)

    for layer, weights_list in new_results.items():
        print(weights_list[0])
        for head, weights in enumerate(weights_list[0]):
            print(weights.size())
            plt.figure(figsize=(10, 8))
            sb.heatmap(weights.numpy(), annot=True, fmt=".2f", cmap="viridis",
                      xticklabels=toks, yticklabels=toks)
            plt.title(f"Layer {layer}, Head {head} - Attention Weights Heatmap")
            plt.xlabel("To")
            plt.ylabel("From")
            plt.savefig('multi_headed_hist_plots_2/heatmap_{}_{}.png'.format(layer, head), format='png', transparent=True,dpi=360, bbox_inches='tight')
            plt.close()

def plot_combined_weights_mean():
    results, toks=collect_example()
    new_results={key: [item['probs'] for item in value] for key, value in results.items()}

    print(toks)

    for layer, weights_list in new_results.items():
        print(weights_list[0].size())
        combined_weights = torch.mean(weights_list[0], dim = 0)
        plt.figure(figsize=(10, 8))
        sb.heatmap(combined_weights.numpy(), annot=True, fmt=".2f", cmap="viridis",
                      xticklabels=toks, yticklabels=toks)
        plt.title(f"Layer {layer}- Attention Weights Heatmap")
        plt.xlabel("To")
        plt.ylabel("From")
        plt.savefig('multi_headed_hist_plots_2/heatmap_{}.png'.format(layer), format='png', transparent=True,dpi=360, bbox_inches='tight')
        plt.close()

def plot_combined_weights_weighted_mean():
    results, toks=collect_example()
    new_results={key: [item['probs'] for item in value] for key, value in results.items()}

    print(toks)

    for layer, weights_list in new_results.items():
        weights_coefficients = torch.nn.Parameter(torch.rand(12))
        weights_coefficients = torch.nn.Softmax(dim = 0)(weights_coefficients)
        combined_weights = torch.sum(weights_list[0] * weights_coefficients.view(12, 1, 1), dim = 0)
        plt.figure(figsize=(10, 8))
        sb.heatmap(combined_weights.detach().numpy(), annot=True, fmt=".2f", cmap="viridis",
                      xticklabels=toks, yticklabels=toks)
        plt.title(f"Layer {layer}- Attention Weights Heatmap")
        plt.xlabel("To")
        plt.ylabel("From")
        plt.savefig('multi_headed_hist_plots_3/heatmap_{}.png'.format(layer), format='png', transparent=True,dpi=360, bbox_inches='tight')
        plt.close()

plot_combined_weights_weighted_mean()