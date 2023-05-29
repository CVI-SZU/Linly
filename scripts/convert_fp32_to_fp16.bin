"""
  This script provides an exmaple to wrap TencentPretrain for generation.
  Given the beginning of a text, language model generates the rest.
"""
import sys
import os
import argparse
import torch
import torch.nn.functional as F

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.embeddings import *
from tencentpretrain.encoders import *
from tencentpretrain.targets import *
from tencentpretrain.utils.constants import *
from tencentpretrain.utils import *
from tencentpretrain.utils.config import load_hyperparam
from tencentpretrain.model_loader import *
from tencentpretrain.opts import model_opts, tokenizer_opts


class GenerateLm(torch.nn.Module):
    def __init__(self, args):
        super(GenerateLm, self).__init__()
        self.embedding = Embedding(args)
        for embedding_name in args.embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.embedding.update(tmp_emb, embedding_name)
        self.encoder = str2encoder[args.encoder](args)
        self.target = Target()
        self.target.update(LmTarget(args, len(args.tokenizer.vocab)), "lm")

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = self.target.lm.output_layer(output)
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    model_opts(parser)

    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path of the input model.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path of the config file.")
    parser.add_argument("--output_model_path", type=str)

    tokenizer_opts(parser)

    args = parser.parse_args()

    args.target = "lm"
    args.batch_size = 1

    args = load_hyperparam(args)

    args.tokenizer = str2tokenizer[args.tokenizer](args)

    model = GenerateLm(args)
    model = load_model(model, args.load_model_path)

    model.half()

    torch.save(model.state_dict(), args.output_model_path)

