"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794

Code copied and modified from Nanochat (https://github.com/karpathy/nanochat)
"""
import random
import time

from jinja2 import Template
import torch
import torch.distributed as dist
import torch.nn.functional as F

from recursive_lm.inference import pack_varlen, forward_logits

# -----------------------------------------------------------------------------
# Prompt rendering utilities

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompt for a language modeling task.
    Notice that we manually trim the context in the template,
    which in some datasets seems to have trailing whitespace (which we don't want).
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # Return two prompts: without and with the continuation
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # Due to the way the data seems to be stored, I think I need to strip in the case of LM here.
    # Otherwise we may get trailing whitespaces in prompt_without (which get absorbed into the next
    # token in prompt_with), meaning we don't get a nice and clean prefix in the token space
    # to detect the final continuation. Tokenizers...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    """
    Find the length of the common prefix or suffix across token sequences
    - direction: 'left' for prefix, 'right' for suffix
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    # Find the first position where the token sequences differ
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def batch_sequences_mc(tokenizer, prompts):
    # In multiple choice, contexts are the same but the continuation is different (common prefix)
    tokens = tokenizer(prompts)
    # figure out the start and end of each continuation
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    # In schema tasks, contexts vary but continuation is the same (common suffix)
    tokens = tokenizer(prompts)
    # figure out the start and end of each context
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    # In LM tasks, we have two prompts: without and with continuation
    tokens = tokenizer(prompts)
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    # we only need the with continuation prompt in the LM task, i.e. batch size of 1
    return [tokens_with], [start_idx], [end_idx]


def _prepare_example(idx, model, tokenizer, data, task_meta):
    """Prepare one example: tokenize prompts and return per-seq spans and metadata."""
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    # Sample few-shot examples (excluding current item)
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    # Render prompts and batch sequences based on task type
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Some models can't forward sequences beyond a certain length (e.g. GPT-2)
    # In these cases, we have to truncate sequences to max length and adjust the indices
    max_tokens = getattr(model, "max_seq_len", None)
    if max_tokens is None:
        max_tokens = getattr(getattr(model, "config", None), "rope_cache_len", None)
    if max_tokens is not None:
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:]) # take the last max_tokens tokens
                new_start_idxs.append(s - num_to_crop) # shift the indices down
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0, "this should never happen right?"
                assert e - num_to_crop >= 0, "this should never happen right?"
            else:
                new_tokens.append(t) # keep unchanged
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    sequences = []
    for i, (t, s, e) in enumerate(zip(tokens, start_idxs, end_idxs)):
        sequences.append({
            "tokens": t,
            "start": s,
            "end": e,
            "choice_idx": i,
        })

    return {
        "task_type": task_type,
        "gold": item.get("gold", None),
        "sequences": sequences,
    }


def _evaluate_batch(prepared_examples, model, device):
    """Evaluate a batch of prepared examples. Returns dict of idx -> correctness."""
    if not prepared_examples:
        return {}

    sequences = []
    seq_meta = []
    example_info = {}
    for ex_idx, prep in prepared_examples:
        seqs = prep["sequences"]
        example_info[ex_idx] = {
            "task_type": prep["task_type"],
            "gold": prep["gold"],
            "scores": [None] * len(seqs),
        }
        for seq in seqs:
            sequences.append(seq["tokens"])
            seq_meta.append({
                "ex_idx": ex_idx,
                "choice_idx": seq["choice_idx"],
                "start": seq["start"],
                "end": seq["end"],
                "task_type": prep["task_type"],
            })

    flat_input, cu_seqlens, position_ids, _lengths, offsets = pack_varlen(sequences, device=device)
    logits = forward_logits(model, flat_input, cu_seqlens, position_ids)

    targets: list[int] = []
    for seq in sequences:
        if len(seq) == 0:
            continue
        targets.extend(seq[1:])
        targets.append(-100)
    targets_t = torch.tensor(targets, dtype=torch.long, device=flat_input.device)

    losses = F.cross_entropy(logits, targets_t, reduction="none", ignore_index=-100)
    predictions = logits.argmax(dim=-1)

    for i, meta in enumerate(seq_meta):
        off = offsets[i]
        si = meta["start"]
        ei = meta["end"]
        if meta["task_type"] == "language_modeling":
            predicted_tokens = predictions[off + si - 1 : off + ei - 1]
            actual_tokens = flat_input[off + si : off + ei]
            correct = torch.all(predicted_tokens == actual_tokens).item()
            example_info[meta["ex_idx"]]["scores"][meta["choice_idx"]] = float(correct)
        else:
            span = losses[off + si - 1 : off + ei - 1]
            example_info[meta["ex_idx"]]["scores"][meta["choice_idx"]] = span.mean().item()

    results = {}
    for ex_idx, info in example_info.items():
        if info["task_type"] == "language_modeling":
            results[ex_idx] = bool(info["scores"][0])
        else:
            pred_idx = info["scores"].index(min(info["scores"]))
            results[ex_idx] = pred_idx == info["gold"]

    return results


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """Evaluate a single example, return True if correct, False otherwise"""
    prep = _prepare_example(idx, model, tokenizer, data, task_meta)
    out = _evaluate_batch([(idx, prep)], model, device)
    return out[idx]

    return is_correct


def evaluate_task(
    model,
    tokenizer,
    data,
    device,
    task_meta,
    *,
    progress_every: int = 200,
    task_label: str | None = None,
    max_batch_tokens: int = 0,
):
    """
    This function is responsible for evaluating one task across many examples.
    It also handles dispatch to all processes if the script is run with torchrun.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    start_time = time.time()
    window_count = 0
    window_start = start_time
    indices = list(range(rank, len(data), world_size))
    total_local = len(indices)
    i = 0
    while i < total_local:
        if max_batch_tokens and max_batch_tokens > 0:
            batch = []
            token_count = 0
            while i < total_local:
                idx = indices[i]
                prep = _prepare_example(idx, model, tokenizer, data, task_meta)
                seq_tokens = sum(len(s["tokens"]) for s in prep["sequences"])
                if batch and token_count + seq_tokens > max_batch_tokens:
                    break
                batch.append((idx, prep))
                token_count += seq_tokens
                i += 1
            batch_results = _evaluate_batch(batch, model, device)
            for ex_idx, is_correct in batch_results.items():
                correct[ex_idx] = float(is_correct)
            batch_size = len(batch)
        else:
            idx = indices[i]
            is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
            correct[idx] = float(is_correct)
            batch_size = 1
            i += 1

        if progress_every and rank == 0:
            window_count += batch_size
            seen = i
            if window_count >= progress_every or seen == total_local:
                now = time.time()
                elapsed = now - window_start
                rate = window_count / elapsed if elapsed > 0 else 0.0
                remaining = total_local - seen
                eta = remaining / rate if rate > 0 else 0.0
                label = task_label or task_meta.get("dataset_uri", "task")
                print(f"{label}: {seen}/{total_local} | {rate:.2f} ex/s | ETA {eta:.1f}s")
                window_count = 0
                window_start = now
    # sync results across all the processes if running distributed
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    # compute the mean
    mean_correct = correct.mean().item()
    return mean_correct
