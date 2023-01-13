import gc
import time
import warnings
from argparse import Namespace
from typing import List, Tuple, Type
import contextlib

import torch
from transformers import AutoTokenizer, BloomForCausalLM, GPT2LMHeadModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import no_init_weights
from transformers import Conv1D


def check_unused(args, defaults, enforce=False):
    for name, default in defaults.items():
        val = getattr(args, name)
        is_default = val is None if default is None else val == default
        if not is_default:
            warnings.warn(f"{'Invalid' if enforce else 'Unexpected'} argument: --{name} (value = {val}, {'setting to' if enforce else 'expected'} {default})")
            if enforce:
                setattr(args, name, default)



def conv1d_init(self, nf, nx, device=None):
    super(Conv1D, self).__init__()
    self.nf = nf
    w = torch.empty(nx, nf, device=device)
    torch.nn.init.normal_(w, std=0.02)
    self.weight = torch.nn.Parameter(w)
    b=torch.empty(nf, device=device)
    torch.nn.init.zeros_(b)
    self.bias = torch.nn.Parameter(b)


Conv1D.__init__ = conv1d_init


@contextlib.contextmanager
def fast_model(classes, device):
    # Avoid multiple slow initializations on cpu.
    default_inits = {cls: cls.__init__ for cls in classes}
    for cls in classes:
        # Same as torch.nn.utils.skip_init, excluding checks
        def init(self,*args, **kwargs):
            default_inits[self.__class__](self, *args, **kwargs, device="meta")
            self.to_empty(device=device)

        cls.__init__ = init

    with no_init_weights():
        yield

    for cls in classes:
        cls.__init__ = default_inits[cls]


class Pipeline:
    def __init__(self, args: Namespace) -> None:
        print("*** Setting up tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.device = args.device

        model_class, config = self.get_config(args)
        is_int8 = args.dtype == torch.int8
        if is_int8:
            check_unused(args, {"device": torch.device("cuda")}, enforce=True)
        torch_dtype = torch.float16 if is_int8 else args.dtype

        print("*** Creating model")
        with fast_model([torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm, Conv1D], self.device):
            self.model = model_class._from_config(config=config, torch_dtype=torch_dtype)
        print("*** Moving to device")
        self.model.to(self.device)
        print("*** Initializing weights")
        # Initialization is ~1000x faster on GPU.
        self.model.init_weights()

        # Int8 can only be obtained by reloading a pretrained model
        if is_int8:
            print("*** Saving model")
            self.model.save_pretrained("tmp")
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            print("*** Reloading model in int8")
            with fast_model([torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm], self.device):
                self.model = model_class.from_pretrained(
                    "tmp",
                    load_in_8bit=True,
                    device_map="auto",
                )


        self.model.eval()

    def get_config(self, args) -> Tuple[Type[PreTrainedModel], PretrainedConfig]:
        config_args = {
            "n_head": args.n_head,
            "n_layer": args.n_layer,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "vocab_size": len(self.tokenizer),
            "use_cache": True,
        }
        if args.model_class.lower() == "bloom":
            check_unused(args, {"attention_type": 1, "n_positions": None})
            config_args["attention_softmax_in_fp32"] = True
            config_args["hidden_size"] = args.hidden_size
            model_class = BloomForCausalLM
        elif args.model_class.lower() == "gpt2":
            config_args["attention_type"] = args.attention_type
            config_args["n_embd"] = args.hidden_size
            config_args["n_positions"] = args.n_positions
            config_args["print_details"] = False
            model_class = GPT2LMHeadModel
        else:
            raise NotImplementedError()

        return model_class, model_class.config_class(**config_args)


    def __call__(self, text: List[str], **generate_kwargs) -> Tuple[List[str], List[int], Tuple[float,float,float]]:
        t0=time.perf_counter()
        input_tokens = self.tokenizer(text, return_tensors="pt", padding=True)

        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(self.device)

        t1=time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(**input_tokens, return_dict_in_generate=True, **generate_kwargs)
        t2=time.perf_counter()

        output_tokens = output.sequences

        input_token_lengths = [x.shape[0] for x in input_tokens.input_ids]
        output_token_lengths = [x.shape[0] for x in output_tokens]
        num_generated_tokens = [o - i for i, o in zip(input_token_lengths, output_token_lengths)]

        output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        t3=time.perf_counter()

        return output_text, num_generated_tokens, (t1-t0, t2-t1, t3-t2)

    def get_num_parameters(self) -> int:
        param_count = 0
        for i in self.model.parameters():
            param_count += i.numel()
        return param_count

class HF_Pipeline(Pipeline):
    def __init__(self, args: Namespace) -> None:
        check_unused(args, {"inject_kernel": False, "cuda_graph": False})
        super().__init__(args)

