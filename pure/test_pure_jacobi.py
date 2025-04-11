import argparse
import os
import sys
sys.path.append("./")

from inference_solver import FlexARInferenceSolver
from PIL import Image
from jacobi_iteration_pure import renew_pipeline_sampler
import torch
import time

import random
import numpy as np

def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

model_path = "nonwhy/PURE"
target_size = 512
text_top_k = 1
temperature = 0.9
guidance_scale = 0.8

max_num_new_tokens = 16
multi_token_init_scheme = 'random' # 'repeat_horizon'
prefix_token_sampler_scheme = 'speculative_jacobi' # 'jacobi', 'speculative_jacobi'

# ******************** Image Generation ********************
inference_solver = FlexARInferenceSolver(
    model_path=model_path,
    precision="bf16",
    target_size=target_size,
)

q1 = "Perceive the degradation level, understand the image content, and restore the high-quality image. <|image|>"
images = [Image.open("test_SR_bicubic/Canon_005_LR4.png")]
qas = [[q1, None]]

inference_solver = renew_pipeline_sampler(
    inference_solver,
    jacobi_loop_interval_l = 3,
    jacobi_loop_interval_r = (target_size // 8)**2 + target_size // 8 - 10,
    max_num_new_tokens = max_num_new_tokens,
    guidance_scale = guidance_scale,
    seed = None,
    multi_token_init_scheme = multi_token_init_scheme,
    do_cfg= True,
    text_top_k=text_top_k,
    prefix_token_sampler_scheme = prefix_token_sampler_scheme,
)

time_start = time.time()
t1 = torch.cuda.Event(enable_timing=True)
t2 = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
t1.record()

generated = inference_solver.generate(
    images=images,
    qas=qas,
    max_gen_len=11776,
    temperature=temperature,
    logits_processor=inference_solver.create_logits_processor(cfg=guidance_scale),
)
t2.record()
torch.cuda.synchronize()

t = t1.elapsed_time(t2) / 1000
time_end = time.time()
print("Time elapsed: ", t, time_end - time_start)

text, new_image = generated[0], generated[1][0]
new_image.save("./Canon_005_test_jacobi.png", "PNG")
print(text)