import argparse
import gc
import json
import math
import os
import shutil

import torch

from transformers import LlamaConfig, LlamaForCausalLM


"""
Sample usage:

```
python llm_model/scripts/convert_llama_tencentpretrain_to_hf.py \
    --tp_model_dir /path/to/downloaded_tencentpretrain_model \
    --input_dir /path/to/downloaded/llama/weights \
    --model_size 7B \
    --model_name chatllama_7b.bin \
    --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import LlamaForCausalLM, LlamaForTokenizer

model = LlamaForCausalLM.from_pretrained("/output/path")
tokenizer = LlamaTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""

INTERMEDIATE_SIZE_MAP = {
    "7B": 11008,
    "13B": 13824,
    "33B": 17920,
    "65B": 22016,
}
NUM_SHARDS = {
    "7B": 1,
    "13B": 2,
    "33B": 4,
    "65B": 8,
}


def compute_intermediate_size(n):
    return int(math.ceil(n * 8 / 3) + 255) // 256 * 256


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)

'''
           model_path=args.output_dir,
            tp_model_dir=args.tp_model_dir,
            tp_model_name=args.tp_model_name,
            input_dir=args.input_dir,
            model_size=args.model_size,
'''
def write_model(model_path, tp_model_dir=None, tp_model_name=None, input_dir=None, model_size=None):
    assert model_path and tp_model_dir and tp_model_name and input_dir and model_size

    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    input_base_path = os.path.join(input_dir, model_size)
    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = NUM_SHARDS[model_size]
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

    # permute for sliced rotary
    def permute(w):
        return w.view(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")

    # '''
    loaded = torch.load(os.path.join(tp_model_dir, tp_model_name+'.bin'))

    param_count = 0
    index_dict = {"weight_map": {}}

    filename = f"pytorch_model-00.bin"
    state_dict = {
        "model.embed_tokens.weight": loaded[
            "embedding.word.embedding.weight"
            # "tok_embeddings.weight"
        ],
        "model.norm.weight": loaded[
            "encoder.layer_norm.weight"
            # "norm.weight"
        ],
        "lm_head.weight": loaded[
            "target.lm.output_layer.weight"
            # "output.weight"
        ],
    }
    for layer_i in range(n_layers):
        state_dict.update({
            f"model.layers.{layer_i}.self_attn.q_proj.weight": permute(loaded[
                                                                           "encoder.transformer." + str(layer_i) + ".self_attn.linear_layers.0.weight"
                                                                           # f"layers.{layer_i}.attention.wq.weight"
                                                                           ]),
            f"model.layers.{layer_i}.self_attn.k_proj.weight": permute(loaded[
                                                                           "encoder.transformer." + str(layer_i) + ".self_attn.linear_layers.1.weight"
                                                                           # f"layers.{layer_i}.attention.wk.weight"
                                                                           ]),
            f"model.layers.{layer_i}.self_attn.v_proj.weight": loaded[
                "encoder.transformer." + str(layer_i) + ".self_attn.linear_layers.2.weight"
                # f"layers.{layer_i}.attention.wv.weight"
                ],
            f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[
                "encoder.transformer." + str(layer_i) + ".self_attn.final_linear.weight"
                # f"layers.{layer_i}.attention.wo.weight"
                ],
            f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[
                "encoder.transformer." + str(layer_i) + ".feed_forward.linear_gate.weight"
                # f"layers.{layer_i}.feed_forward.w1.weight"
                ],
            f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[
                "encoder.transformer." + str(layer_i) + ".feed_forward.linear_2.weight"
                # f"layers.{layer_i}.feed_forward.w2.weight"
                ],
            f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[
                "encoder.transformer." + str(layer_i) + ".feed_forward.linear_1.weight"
                # f"layers.{layer_i}.feed_forward.w3.weight"
                ],
            f"model.layers.{layer_i}.input_layernorm.weight": loaded[
                "encoder.transformer." + str(layer_i) + ".layer_norm_1.weight"
                # f"layers.{layer_i}.attention_norm.weight"
                ],
            f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[
                "encoder.transformer." + str(layer_i) + ".layer_norm_2.weight"
                # f"layers.{layer_i}.ffn_norm.weight"
                ],
        })
        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    config = LlamaConfig(
        hidden_size=dim,
        intermediate_size=compute_intermediate_size(dim),
        num_attention_heads=params["n_heads"],
        num_hidden_layers=params["n_layers"],
        rms_norm_eps=params["norm_eps"],
    )
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()
    # '''

    print("Loading the checkpoint in a Llama model.")
    model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # Avoid saving this as part of the config.
    del model.config._name_or_path

    print("Saving in the Transformers format.")
    model.save_pretrained(model_path)
    shutil.rmtree(tmp_model_path)


def write_tokenizer(tokenizer_path, input_tokenizer_path):
    print(f"Fetching the tokenizer from {input_tokenizer_path}.")
    os.makedirs(tokenizer_path, exist_ok=True)
    write_json({}, os.path.join(tokenizer_path, "special_tokens_map.json"))
    write_json(
        {
            "bos_token": "",
            "eos_token": "",
            "model_max_length": int(1e30),
            "tokenizer_class": "LlamaTokenizer",
            "unk_token": "",
        },
        os.path.join(tokenizer_path, "tokenizer_config.json"),
    )
    shutil.copyfile(input_tokenizer_path, os.path.join(tokenizer_path, "tokenizer.model"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tp_model_dir",
        help="Location of tencentpretrain LLaMA weights, which contains model folders",
    )
    parser.add_argument(
        "--tp_model_name",
        help="tencentpretrain model name",
    )
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        choices=["7B", "13B", "33B", "65B", "tokenizer_only"],
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    args = parser.parse_args()
    if args.model_size != 'tokenizer_only':
        write_model(
            model_path=args.output_dir,
            tp_model_dir=args.tp_model_dir,
            tp_model_name=args.tp_model_name,
            input_dir=args.input_dir,
            model_size=args.model_size,
        )
    write_tokenizer(
        tokenizer_path=args.output_dir,
        input_tokenizer_path=os.path.join(args.input_dir, "tokenizer.model"),
    )


if __name__ == "__main__":
    main()
