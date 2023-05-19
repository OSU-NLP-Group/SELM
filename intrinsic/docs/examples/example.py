import torch
import transformers

import intrinsic

device = torch.device(0)

gpt2 = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
gpt2 = gpt2.to(device)

intrinsic.intrinsic_dimension_said(gpt2, 1000, "", set(), "fastfood", device=device)

tokens = torch.tensor([1024, 1025, 1026], device=device)

out = gpt2(input_ids=tokens)

loss = torch.sum(out.logits)

loss.backward()

breakpoint()
