# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import contextlib
import os
import random

import kagglehub as kagglehub
import numpy as np
import torch

from gemma import config
from gemma import model as gemma_model


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


def main(args):
    # Construct the model config.
    model_config = config.get_model_config(args.variant)
    model_config.dtype = "float32" if args.device == "cpu" else "float16"
    model_config.quant = args.quant
    model_config.tokenizer = args.tokenizer

    # Seed random.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create the model and load the weights.
    device = torch.device(args.device)
    with _set_default_tensor_type(model_config.get_dtype()):
        model = gemma_model.GemmaForCausalLM(model_config)
        model.load_weights(args.ckpt)
        model = model.to(device).eval()
    print("Model loading done")

    # Generate the response.
    result = model.generate(args.prompt, device, output_len=args.output_len)

    # Print the prompts and results.
    print('======================================')
    print(f'PROMPT: {args.prompt}')
    print(f'RESULT: {result}')
    print('======================================')


class Arguments:
    ckpt: str  # Replace with your desired path or value
    variant: str  # Can be either "2b" or "7b"
    device: str  # Can be either "cpu" or "cuda"
    output_len: int
    seed: int
    quant: bool  # Set to True if you want to enable quantization
    prompt: str  # e.g. "The meaning of life is"
    tokenizer: str # path to tokenizer

    def __init__(self, ckpt, variant="2b", device="cpu", output_len=100, seed=47, quant=False, prompt="The meaning of life is", tokenizer='tokenizer.model'):

        # You can add checks for 'choices' if needed
        if variant not in ['2b', '2b-it', '7b', '7b-it', '7b-quant', '7b-it-quant']:
            raise ValueError("Invalid value for 'variant'. Choose between '2b' and '7b'")
        if device not in ["cpu", "cuda"]:
            raise ValueError("Invalid value for 'device'. Choose between 'cpu' and 'cuda'")

        self.ckpt = ckpt
        self.variant = variant
        self.device = device
        self.output_len = output_len
        self.seed = seed
        self.quant = quant
        self.prompt = prompt
        self.tokenizer = tokenizer


if __name__ == "__main__":
    params = Arguments(ckpt='', variant='2b', device='cpu', output_len=100, seed=47, quant=False, prompt="The meaning of life is", tokenizer='tokenizer.model')

    # Load model weights
    weights_dir = kagglehub.model_download(f'google/gemma/pyTorch/{params.variant}')

    # Ensure that the tokenizer is present
    tokenizer_path = os.path.join(weights_dir, 'tokenizer.model')
    print(f"tokenizer_path: {tokenizer_path}")
    assert os.path.isfile(tokenizer_path), 'Tokenizer not found!'
    params.tokenizer = tokenizer_path

    # Ensure that the checkpoint is present
    ckpt_path = os.path.join(weights_dir, f'gemma-{params.variant}.ckpt')
    print(f"ckpt_path: {ckpt_path}")
    assert os.path.isfile(ckpt_path), 'PyTorch checkpoint not found!'
    params.ckpt = ckpt_path

    main(params)
