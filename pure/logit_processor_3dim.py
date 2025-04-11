import json
import copy
import numpy as np
from functools import cached_property
import math
from typing import Optional, Tuple, Union, List, Dict, Sequence, Any
import random
import torch
from torch import nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.generation.logits_process import (
    LogitsProcessor,
    TopKLogitsWarper,
)
from transformers.generation.logits_process import LogitsWarper

from transformers.generation import PrefixConstrainedLogitsProcessor

import time

def check_eol_in_multitokens(tokenlen, new_pred_tokenlen, line_len):
    L, R = (tokenlen + 1), (tokenlen + new_pred_tokenlen)
    check_interval_l = L // line_len + 1 if L % line_len != 0 else L // line_len
    check_interval_r = R // line_len
    return (check_interval_l <= check_interval_r)

def get_eol_in_multitokens(logits, eol_cls, tokenlen, new_pred_tokenlen, line_len, min_dtype=-math.inf):
    logits_forced_eol = logits.clone()
    L, R = (tokenlen + 1), (tokenlen + new_pred_tokenlen)
    check_interval_l = L // line_len + 1 if L % line_len != 0 else L // line_len
    check_interval_r = R // line_len
    eol_position_ids = [ 
        line_len * multi_num - (tokenlen + 1) for multi_num in range(check_interval_l, check_interval_r + 1)
    ]
    for i in eol_position_ids:
        logits_forced_eol[..., i, :] = min_dtype
        logits_forced_eol[..., i, eol_cls] = 0

    return logits_forced_eol, eol_position_ids

class MultiTokensVLLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        image_start_token_id=None,
        image_end_token_id=None,
        image_next_line_token_id=None,
        patch_size=None,
        voc_size=None,
        device = "cuda",
    ):
        self.image_start_token_id = image_start_token_id # 8197
        self.image_end_token_id = image_end_token_id # 8196
        self.image_next_line_token_id = image_next_line_token_id # 8803
        self.image_start_token_id_index = None
        self.patch_size = patch_size
        self.h_latent_dim = None
        self.w_latent_dim = None

        self.vocab_list = [i for i in range(voc_size)]
        self.image_token_list = [i for i in range(4, 8195 + 1)]
        self.suppress_tokens = torch.tensor(
            [x for x in self.vocab_list if x not in self.image_token_list], device=device
        )

        self.vocab_tensor = torch.arange(voc_size, device=device)
        self.suppress_token_mask = torch.isin(self.vocab_tensor, self.suppress_tokens) # not [   4,    5,    6,  ..., 8193, 8194, 8195]
        self.new_line_force_token_mask = torch.isin(
            self.vocab_tensor, torch.tensor([self.image_next_line_token_id], device=device)
        )
        self.eos_image_force_token_mask = torch.isin(
            self.vocab_tensor, torch.tensor([self.image_end_token_id], device=device)
        )

        self.flag = False
        self.num_image_start_tokens = None
        self.num_image_end_tokens = None

    # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        self.num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        self.num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        if self.num_image_start_tokens == self.num_image_end_tokens:
            self.h_latent_dim, self.w_latent_dim = None, None
            self.image_start_token_id_index = None
            return scores

        elif self.num_image_start_tokens == self.num_image_end_tokens + 1:
            if self.image_start_token_id_index is None:
                self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0]
                self.image_start_token_id_index = torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item()

            new_logit_token_len = scores.shape[-2] if len(scores.shape) >= 3 else 1

            new_token_num = len(input_ids[0][self.image_start_token_id_index + 1 :])
            if new_token_num >= 2:

                pad_eol_len = 1 # TODO: to check

                if self.h_latent_dim is None or self.w_latent_dim is None:
                    h_grids, w_grids = (
                        input_ids[0][self.image_start_token_id_index + 1] - 8804,
                        input_ids[0][self.image_start_token_id_index + 2] - 8804,
                    )
                    self.h_latent_dim, self.w_latent_dim = h_grids * 4, w_grids * 4
                    print('self.h_latent_dim, self.w_latent_dim', self.h_latent_dim, self.w_latent_dim)

                tokens = input_ids[0][self.image_start_token_id_index + 3 :]

                is_new_seq_ids_containing_end_of_line = check_eol_in_multitokens(
                    len(tokens), new_logit_token_len, self.w_latent_dim + pad_eol_len
                )
                is_new_seq_ids_containing_end_of_img = check_eol_in_multitokens(
                    len(tokens), new_logit_token_len, 
                    (self.w_latent_dim + pad_eol_len) * self.h_latent_dim + pad_eol_len
                )

                # TODO: is_pre_seq_containing_end_of_img:
                scores = torch.where(
                    self.suppress_token_mask.to(scores.device), 
                    -float("inf"), 
                    scores
                )

                # containing ONE end-of-line
                if is_new_seq_ids_containing_end_of_line:

                    scores, eol_position_ids = get_eol_in_multitokens(
                        scores, self.image_next_line_token_id, len(tokens), 
                        new_logit_token_len, self.w_latent_dim + pad_eol_len
                    )
                
                # containing ONE end-of-image
                if is_new_seq_ids_containing_end_of_img:
                    scores, eol_position_ids = get_eol_in_multitokens(
                        scores, self.image_end_token_id, len(tokens), 
                        new_logit_token_len, 
                        (self.w_latent_dim + pad_eol_len) * self.h_latent_dim + pad_eol_len,
                    )
                    
                return scores
        else:
            print(f"Something wrong in the decoding process. MultiTokensVLLogitsProcessor. \
                  st: id {torch.where(input_ids[0] == self.image_start_token_id)} num {self.num_image_start_tokens} \
                  ed: id {torch.where(input_ids[0] == self.image_end_token_id)} num {self.num_image_end_tokens} \
                  input_ids.shape {input_ids.shape} scores.shape {scores.shape} "
            )

        return scores


class MultiTokensInterleavedTopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements. Often used together
    with [`TemperatureLogitsWarper`] and [`TopPLogitsWarper`].
    """

    def __init__(
        self,
        image_top_k: int = 2000,
        text_top_k: int = 1,
        image_start_token_id=None,
        image_end_token_id=None,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
        min_top_k: int = 1,    # 最小top-k值
        max_top_k: int = 2000,  # 最大top-k值
    ):
        if not isinstance(text_top_k, int) or text_top_k <= 0:
            raise ValueError(f"`text_top_k` has to be a strictly positive integer, but is {text_top_k}")
        if not isinstance(image_top_k, int) or text_top_k <= 0:
            raise ValueError(f"`image_top_k` has to be a strictly positive integer, but is {image_top_k}")

        # self.image_top_k = max(image_top_k, min_tokens_to_keep)
        self.text_top_k = max(text_top_k, min_tokens_to_keep)
        self.filter_value = filter_value

        self.min_top_k = min_top_k
        self.max_top_k = max_top_k

        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id

        self.flag = False
        self.num_image_start_tokens = None
        self.num_image_end_tokens = None

    # @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        self.num_image_start_tokens = (input_ids[0] == self.image_start_token_id).sum()
        self.num_image_end_tokens = (input_ids[0] == self.image_end_token_id).sum()

        probs = torch.softmax(scores, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()

        if self.num_image_start_tokens == self.num_image_end_tokens + 1:
            current_pos = len(input_ids[0]) - (torch.where(input_ids[0] == self.image_start_token_id)[0][-1].item() + 1)
            if current_pos == 1:
                h_grids = (input_ids[0][-1] - 8804) * 4
                self.current_image_size = [h_grids, None]
            elif current_pos == 2:
                w_grids = (input_ids[0][-1] - 8804) * 4
                self.current_image_size[1] = w_grids
                self.image_entropies = []
            elif current_pos > 2 and self.current_image_size is not None:
                if (current_pos - 3) % (self.current_image_size[1] + 1) != self.current_image_size[1]:
                    self.image_entropies.append(entropy.item())

        if self.num_image_start_tokens == self.num_image_end_tokens + 1:
            # top_k = min(self.image_top_k, scores.size(-1))
            top_k = self._calculate_dynamic_top_k(entropy.item())
            print(f"Entropy: {entropy.item():.4f}, Dynamic top-k: {top_k}")
        else:
            top_k = min(self.text_top_k, scores.size(-1))  # Safety check
        
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]

        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed

    def _calculate_dynamic_top_k(self, entropy: float) -> int:
            entropy = max(0, min(entropy, 10))
            x = entropy - 5
            sigmoid_value = 1 / (1 + np.exp(-x))
            dynamic_k = self.min_top_k + (self.max_top_k - self.min_top_k) * sigmoid_value
            dynamic_k = int(round(dynamic_k))
            dynamic_k = max(self.min_top_k, min(dynamic_k, self.max_top_k))
            return dynamic_k


class AllowOnlyTokensAtRelativeOffsetLogitsProcessor3d(LogitsProcessor):
    r"""
    [`AllowOnlyTokensAtRelativeOffsetLogitsProcessor`] suppresses the logits of tokens aside from a specific set of tokens
    that can be generated at a relative offset from a trigger token (e.g. begin image token). If `exclusive` is set to
    `True`, the set of tokens allowed at this offset will not be allowed anywhere else. This is useful for enforcing
    multimodal generation constraints with begin and end marker tokens.

    Originally created for [Chameleon](https://huggingface.co/docs/transformers/model_doc/chameleon).

    Args:
        trigger_token_id (`int`):
            The token id that triggers the offset check.
        allowed_token_ids (`List[int]`):
            The list of token ids that are allowed at the specified offset.
        offset (`int`):
            The relative offset from the trigger token.
        exclusive (`bool`, *optional*, defaults to `False`):
            If `True`, the set of tokens allowed at this offset will not be allowed anywhere else.
        device (`str`, *optional*, defaults to `cpu`):
            The device to allocate the util tensor on.
    """

    def __init__(
        self,
        trigger_token_id: int,
        allowed_token_ids: List[int],
        offset: int,
        exclusive: bool = False,
        device: str = "cpu",
    ):
        self.trigger_token_id = trigger_token_id
        self.allowed_token_ids = torch.tensor(allowed_token_ids, device=device)
        self.offset = offset
        self.exclusive = exclusive

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] < self.offset and not self.exclusive:
            return scores

        disallowed_tokens_mask = torch.ones_like(scores, dtype=torch.bool)
        disallowed_tokens_mask[..., self.allowed_token_ids] = False

        if input_ids.shape[1] < self.offset:
            return scores.masked_fill(~disallowed_tokens_mask, torch.finfo(scores.dtype).min)

        trigger_positions = (input_ids[..., -self.offset] == self.trigger_token_id).unsqueeze(-1)

        if self.exclusive:
            return scores.masked_fill(~(disallowed_tokens_mask ^ trigger_positions), torch.finfo(scores.dtype).min)
        return scores.masked_fill(disallowed_tokens_mask & trigger_positions, torch.finfo(scores.dtype).min)

class SuppressTokensInIndexRangeLogitsProcessor3d(LogitsProcessor):
    r"""
    [`SuppressTokensInIndexRangeLogitsProcessor`] supresses a list of tokens from `start_index` to `end_index` (exclusive)

    Args:
        suppress_tokens (`List[int]`):
            List of token ids to suppress during generation.
        start_index (`int`):
            The index at which to start suppressing tokens.
        end_index (`int`, *optional*):
            The index at which to end suppressing tokens. If `None`, it will suppress tokens indefinitely.
        device (`str`, *optional*, defaults to `"cpu"`):
            The device to allocate the tensors.
    """

    def __init__(
        self, suppress_tokens: List[int], start_index: int, end_index: Optional[int] = None, device: str = "cpu"
    ):
        self.suppress_tokens = torch.tensor(suppress_tokens, device=device)
        self.start_index = start_index
        self.end_index = end_index if end_index is not None else math.inf

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        current_index = input_ids.shape[1]
        if self.start_index > current_index or current_index > self.end_index:
            return scores
        suppress_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        suppress_tokens_mask[..., self.suppress_tokens] = True
        return scores.masked_fill(suppress_tokens_mask, torch.finfo(scores.dtype).min)

class AllowOnlyTokensInRelativeWindowLogitsProcessor3d(LogitsProcessor):
    r"""
    [`AllowOnlyTokensInRelativeWindowLogitsProcessor`] suppresses the logits of tokens aside from a specific set of tokens
    that can be generated at a relative window from a trigger token (e.g. begin image token). If `exclusive` is set to
    `True`, the set of tokens allowed at this window will not be allowed anywhere else. This is useful for enforcing
    multimodal generation constraints.

    Originally created for [Chameleon](https://huggingface.co/docs/transformers/model_doc/chameleon).

    Args:
        trigger_token_id (`int`):
            The token id that triggers the window check.
        allowed_token_ids (`List[int]`):
            The list of token ids that are allowed at the specified relative window.
        window_width (`int`):
            The window_width of the window from the trigger token.
        exclusive (`bool`, *optional*, defaults to `False`):
            If `True`, the set of tokens allowed at this window will not be allowed anywhere else.
        device (`str`, *optional*, defaults to `cpu`):
            The device to allocate the util tensor on.
    """

    def __init__(
        self,
        trigger_token_id: int,
        allowed_token_ids: List[int],
        window_width: int,
        exclusive: bool = False,
        device: str = "cpu",
    ):
        self.trigger_token_id = trigger_token_id
        self.allowed_token_ids = torch.tensor(allowed_token_ids, device=device).unsqueeze(0)
        self.window_width = window_width
        self.exclusive = exclusive

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        window_width = min(self.window_width, input_ids.shape[1])
        trigger_positions = (input_ids[..., -window_width:] == self.trigger_token_id).any(dim=1).unsqueeze(-1)

        disallowed_tokens_mask = torch.ones_like(scores, dtype=torch.bool)
        disallowed_tokens_mask[..., self.allowed_token_ids] = False

        if self.exclusive:
            return scores.masked_fill(
                ~(disallowed_tokens_mask ^ trigger_positions),
                torch.finfo(scores.dtype).min,
            )
        return scores.masked_fill(
            disallowed_tokens_mask & trigger_positions,
            torch.finfo(scores.dtype).min,
        )

class SuppressTokensAtBeginLogitsProcessor3d(SuppressTokensInIndexRangeLogitsProcessor3d):
    def __init__(self, begin_suppress_tokens, begin_index, device: str = "cpu"):
        super().__init__(begin_suppress_tokens, begin_index, begin_index + 1, device=device)
        self.begin_index = begin_index

    def set_begin_index(self, begin_index):
        self.start_index = begin_index
        self.end_index = begin_index + 1
        # Keeping this here for backwards compatibility
        self.begin_index = begin_index

class SuppressTokensLogitsProcessor3d(SuppressTokensInIndexRangeLogitsProcessor3d):
    def __init__(self, suppress_tokens, device: str = "cpu"):
        super().__init__(suppress_tokens, 0, device=device)

class TopPLogitsWarper3d(LogitsProcessor):
    """
    [`LogitsProcessor`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.
    Often used together with [`TemperatureLogitsWarper`] and [`TopKLogitsWarper`].

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(1)
    >>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3 | < 4 (left-hand pointer) ;
    <BLANKLINE>
    <BLANKLINE>

    >>> # With `top_p` sampling, the output gets restricted to high-probability tokens.
    >>> # Pro tip: In practice, LLMs use `top_p` in the 0.9-0.95 range.
    >>> outputs = model.generate(**inputs, do_sample=True, top_p=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        # indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed


def get_double_cfg_input_ids(input_ids, neg_input_ids, pad_category):
    batchsize, prefill_num = input_ids.shape
    
    neg_prefill_num = neg_input_ids.shape[1]

    batchsize_cfg = 2 * batchsize
    max_prefill_num = max(prefill_num, neg_prefill_num)

    new_neg_input_ids = torch.full(
        (batchsize_cfg, max_prefill_num),
        pad_category,
        dtype=input_ids.dtype,
        device=input_ids.device
    )

    new_neg_input_ids[:batchsize, -input_ids.shape[1]:] = input_ids
    new_neg_input_ids[batchsize:, -neg_input_ids.shape[1]:] = neg_input_ids

    return new_neg_input_ids

def multinomial_token_sample(logits, generator=None):
    probs = nn.functional.softmax(logits, dim=-1)
    # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
    probs_shape = None
    if len(probs.shape) >= 3:
        probs_shape = probs.shape
        probs = probs.flatten(0, len(probs_shape)-2)

    next_tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
    if probs_shape is not None:
        next_tokens = next_tokens.reshape(probs_shape[:-1])
        probs = probs.reshape(probs_shape)
    
    return next_tokens, probs

class SequenceSegmentDecomposer(LogitsProcessor):
    def __init__(self, sublogit_processors: List[LogitsProcessor], do_sample: bool = True, seed=None, fix_logits=True):
        self.sublogit_processors = sublogit_processors
        self.do_sample = do_sample
        self.generator = torch.Generator(seed) if seed is not None else None
        self.fix_logits = fix_logits
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = []

        # a = time.time()

        while len(scores.shape) < 3:
            scores = scores.unsqueeze(1)

        B, L, V = scores.shape

        input_ids_candidates = input_ids.new_empty(B, input_ids.shape[1] + scores.shape[1])
        input_ids_candidates[:, :input_ids.shape[1]] = input_ids

        for seq_id in range(L):
            next_score = scores[:, seq_id, :]
            input_ids_cum = input_ids_candidates[:, :input_ids.shape[1] + seq_id]
            for sublogit_processor in self.sublogit_processors:
                next_score = sublogit_processor(input_ids_cum, next_score)
            
            if self.do_sample:
                next_token, _ = multinomial_token_sample(
                    next_score,
                    self.generator,
                )
            else:
                next_token = torch.argmax(next_score, dim=-1)
            
            while len(next_token.shape) < 2:
                next_token = next_token.unsqueeze(1)
            
            input_ids_candidates[:, input_ids.shape[1] + seq_id] = next_token.squeeze(1)

            if self.fix_logits:
                # the beam search should condition on fixed tokens, cannot sample the logits again
                # so mask all logits in `next_score` to -inf instead of `next_token`
                new_next_score = torch.full_like(next_score, -math.inf)
                # sampled_logits = torch.gather(next_score, 1, next_token)
                # new_next_score.scatter_(1, next_token, sampled_logits)
                new_next_score.scatter_(1, next_token, 0) # reduce='multiply'
            else:
                new_next_score = next_score

            scores_processed.append(new_next_score)
        
        scores_processed = torch.stack(scores_processed, dim=1)
        del input_ids_candidates
        # print(f"Time: {time.time() - a}")
        return scores_processed

def gather_from_split_tensors(
    tensor_list, 
    indexes, 
    dim=0,
    prefilled_length=0,
    device='cuda',
):
    """
        indexes: tensor of indexes
    """
    
    cum_lengths = [prefilled_length, ] + [t.shape[dim] for t in tensor_list]
    cum_lengths = torch.tensor(cum_lengths, device=device).cumsum(0)
    gathered_tensor = []
    for i, t in enumerate(tensor_list):
        # print(cum_lengths, indexes, i, t.shape)
        relative_indexes = indexes[
            (indexes >= cum_lengths[i]) & (indexes < cum_lengths[i + 1])
        ] - cum_lengths[i]

        selected_tensor = torch.index_select(t, dim=dim, index=relative_indexes, out=None)

        gathered_tensor.append(selected_tensor)
    
    gathered_tensor = torch.cat(gathered_tensor, dim=dim)
    return gathered_tensor