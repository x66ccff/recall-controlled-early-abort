#!/usr/bin/env python

import argparse
import bisect
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
FIXED_SYSTEM = os.environ.get(
    "AGENT_EARLY_ABORT_FIXED_SYSTEM", "You are a helpful assistant.")
FIXED_DATE = "26 Jul 2024"
ENABLE_THINKING_ENV = os.environ.get("AGENT_EARLY_ABORT_ENABLE_THINKING")
ROLE_MAP = {"human": "user", "gpt": "assistant"}
FORCE_ROUNDWISE_ENV = os.environ.get("AGENT_EARLY_ABORT_FORCE_ROUNDWISE_FEATURES")

def thinking_disabled():
    return (ENABLE_THINKING_ENV is not None
            and ENABLE_THINKING_ENV.lower() in {"0", "false", "no", "off"})

def force_roundwise():
    return (FORCE_ROUNDWISE_ENV is not None
            and FORCE_ROUNDWISE_ENV.lower() not in {"0", "false", "no", "off"})

def chat_template_kwargs():
    kwargs = {"date_string": FIXED_DATE}
    if ENABLE_THINKING_ENV is not None:
        kwargs["enable_thinking"] = not thinking_disabled()
    return kwargs

def merge_msgs(conversation):
    msgs = [{"role": "system", "content": FIXED_SYSTEM}]
    for m in conversation:
        role = ROLE_MAP[m["from"]]
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n" + m["value"]
        else:
            msgs.append({"role": role, "content": m["value"]})
    return msgs

def build_prompt(tokenizer, conversation):

    return tokenizer.apply_chat_template(
        merge_msgs(conversation), tokenize=False,
        add_generation_prompt=True, **chat_template_kwargs())

def get_env_client_cls(env_name):
    if env_name == "textcraft":
        from agentenv.envs import TextCraftEnvClient
        return TextCraftEnvClient
    if env_name == "webshop":
        from agentenv.envs.webshop import WebshopEnvClient
        return WebshopEnvClient
    if env_name == "alfworld":
        from agentenv.envs.alfworld import AlfWorldEnvClient
        return AlfWorldEnvClient
    raise ValueError(f"unsupported env: {env_name}")

def get_conversation_start(args):
    env_client_cls = get_env_client_cls(args.env)
    cs = getattr(env_client_cls, "conversation_start", None)
    if cs:
        return [dict(m) for m in cs]
    adapter_cls = getattr(env_client_cls, "adapter_cls", None)
    if adapter_cls is not None:
        cs_dict = getattr(adapter_cls, "conversation_start_dict", None)
        if cs_dict:
            try:
                from agentenv.controller.types import ActionFormat
                return [dict(m) for m in cs_dict[ActionFormat.REACT]]
            except Exception:
                return [dict(m) for m in next(iter(cs_dict.values()))]
    if not args.server_base:
        raise RuntimeError(
            "conversation_start is not a class attribute; pass --server_base so a client can be instantiated")
    c = env_client_cls(
        env_server_base=args.server_base.split(",")[0].rstrip("/"),
        data_len=1, timeout=300)
    return [dict(m) for m in c.conversation_start]

def rebuild_conv(conv_start, rec):
    conv = [dict(m) for m in conv_start]
    conv.append({"from": "human", "value": rec["init_obs"].strip()})
    for rd in rec["rounds"]:
        conv.append({"from": "gpt", "value": rd["gen_text"].strip()})
        conv.append({"from": "human", "value": rd["obs_text"].strip()})
    return conv

def char_to_token(offset_ends, char_pos):

    i = bisect.bisect_right(offset_ends, char_pos) - 1
    if i < 0:
        raise ValueError(f"char_pos={char_pos} has no preceding token")
    return i

def _prefix_end(full_text, prefix, round_idx):
    if full_text.startswith(prefix):
        return len(prefix)

    p = prefix.rstrip()
    assert full_text.startswith(p), f"prefix alignment failed at round={round_idx}"
    return len(p)

def locate_anchors(tokenizer, conv_start, rec):

    conv = rebuild_conv(conv_start, rec)
    full_text = tokenizer.apply_chat_template(
        merge_msgs(conv), tokenize=False,
        add_generation_prompt=False, **chat_template_kwargs())

    anchors = []

    base = len(conv_start) + 1
    for t, rd in enumerate(rec["rounds"]):
        prefix = build_prompt(tokenizer, conv[: base + 2 * t])
        pre_char = _prefix_end(full_text, prefix, t)

        if t + 1 < len(rec["rounds"]):
            next_prefix = build_prompt(tokenizer, conv[: base + 2 * (t + 1)])
            next_pre_char = _prefix_end(full_text, next_prefix, t + 1)
            obs = rec["rounds"][t]["obs_text"].strip()
            if obs:
                obs_char = full_text.rfind(obs, pre_char, next_pre_char)
                assert obs_char != -1, (
                    f"obs_text alignment failed task={rec['task_idx']} "
                    f"k={rec['rollout_k']} round={t}")
                end_char = full_text.rfind("<|im_end|>", pre_char, obs_char)
                post_char = end_char if end_char != -1 else obs_char
            else:
                post_char = next_pre_char
        else:
            end_marker = "<|im_end|>"
            end_char = full_text.find(end_marker, pre_char)
            post_char = end_char if end_char != -1 else len(full_text)

        anchors.append((t, "pre_gen", pre_char))
        anchors.append((t, "post_gen", post_char))
    return full_text, anchors

def _looks_err(obs_text):
    if not isinstance(obs_text, str):
        return False
    low = obs_text.lower()
    return any(k in low for k in ("error", "invalid", "fail", "not found", "cannot"))

def _surface_meta(rec, round_lps, t, typ, ti):

    n_done = t + 1 if typ == "post_gen" else t
    prev = [lp for rr in range(n_done) for lp in round_lps[rr]]
    prefix_mean_lp = float(np.mean(prev)) if prev else 0.0
    cur_round_lp = (float(np.mean(round_lps[n_done - 1]))
                    if n_done >= 1 and round_lps[n_done - 1] else 0.0)

    n_err = sum(1 for rr in range(t)
                if _looks_err(rec["rounds"][rr].get("obs_text", "")))

    return {
        "task_idx": rec["task_idx"],
        "rollout_k": rec["rollout_k"],
        "round": t,
        "anchor": typ,
        "token_idx": ti,
        "success": rec["success"],
        "n_rounds": rec["n_rounds"],
        "hit_max_rounds": rec["hit_max_rounds"],
        "prefix_len": int(ti),
        "prefix_mean_lp": prefix_mean_lp,
        "cur_round_lp": cur_round_lp,
        "n_err": n_err,
    }

@torch.inference_mode()
def extract_one_roundwise(model, tokenizer, conv_start, rec, layer_ids, device):

    conv = [dict(m) for m in conv_start]
    conv.append({"from": "human", "value": rec["init_obs"].strip()})
    round_lps = [r["token_logprobs"] or [] for r in rec["rounds"]]
    rows, feats, mismatch = [], [], 0

    for t, rd in enumerate(rec["rounds"]):
        prompt = build_prompt(tokenizer, conv)
        gen = rd["gen_text"]
        full_text = prompt + gen
        enc = tokenizer(full_text, return_offsets_mapping=True,
                        add_special_tokens=False, return_tensors=None)
        ids = enc["input_ids"]
        offset_ends = [e for (_, e) in enc["offset_mapping"]]
        pre_ti = char_to_token(offset_ends, len(prompt))
        post_ti = char_to_token(offset_ends, len(full_text))
        if abs((post_ti - pre_ti) - rd["n_gen_tokens"]) > 2:
            mismatch += 1

        input_ids = torch.tensor([ids], device=device)
        out = model(input_ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        for typ, ti in (("pre_gen", pre_ti), ("post_gen", post_ti)):
            vec = torch.stack([hs[l][0, ti] for l in layer_ids])
            feats.append(vec.to(torch.float16).cpu().numpy())
            rows.append(_surface_meta(rec, round_lps, t, typ, ti))
        del out, hs

        conv.append({"from": "gpt", "value": gen.strip()})
        conv.append({"from": "human", "value": rd["obs_text"].strip()})

    return rows, np.stack(feats), mismatch

@torch.inference_mode()
def extract_one(model, tokenizer, conv_start, rec, layer_ids, device):
    if force_roundwise():
        return extract_one_roundwise(
            model, tokenizer, conv_start, rec, layer_ids, device)
    try:
        full_text, anchors = locate_anchors(tokenizer, conv_start, rec)
    except AssertionError:
        if not thinking_disabled():
            raise
        return extract_one_roundwise(
            model, tokenizer, conv_start, rec, layer_ids, device)
    enc = tokenizer(full_text, return_offsets_mapping=True,
                    add_special_tokens=False, return_tensors=None)
    ids = enc["input_ids"]
    offset_ends = [e for (_, e) in enc["offset_mapping"]]

    tok_anchors, mismatch = [], 0
    for (t, typ, cpos) in anchors:
        tok_anchors.append((t, typ, char_to_token(offset_ends, cpos)))
    by_round = {}
    for t, typ, ti in tok_anchors:
        by_round.setdefault(t, {})[typ] = ti
    for t, d in by_round.items():
        n_gen = d["post_gen"] - d["pre_gen"]
        if abs(n_gen - rec["rounds"][t]["n_gen_tokens"]) > 2:
            mismatch += 1

    input_ids = torch.tensor([ids], device=device)
    out = model(input_ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states

    round_lps = [r["token_logprobs"] or [] for r in rec["rounds"]]

    rows, feats = [], []
    for (t, typ, ti) in tok_anchors:
        vec = torch.stack([hs[l][0, ti] for l in layer_ids])
        feats.append(vec.to(torch.float16).cpu().numpy())
        rows.append(_surface_meta(rec, round_lps, t, typ, ti))
    del out, hs
    return rows, np.stack(feats), mismatch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--env", choices=("textcraft", "webshop", "alfworld"),
        default="textcraft")
    ap.add_argument("--server_base", default="")
    ap.add_argument("--layers", default="stride2",
                    help="'all' | 'strideN' | comma-separated layer ids (0=embedding)")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip_anchor_failures", action="store_true",
                    help="skip rollout records whose generated text cannot be re-aligned")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True).to(args.device).eval()
    except ValueError as exc:
        if "Qwen3VLConfig" not in str(exc):
            raise
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True).to(args.device).eval()
    model_config = getattr(model.config, "text_config", model.config)
    n_layers = model_config.num_hidden_layers

    if args.layers == "all":
        layer_ids = list(range(n_layers + 1))
    elif args.layers.startswith("stride"):
        s = int(args.layers[len("stride"):])
        layer_ids = sorted(set(list(range(0, n_layers + 1, s)) + [n_layers]))
    else:
        layer_ids = [int(x) for x in args.layers.split(",")]
    print(f"[feat] extracted layers: {layer_ids}", flush=True)

    conv_start = get_conversation_start(args)

    recs = [json.loads(l) for l in open(args.rollouts)]
    recs = [r for r in recs if r["n_rounds"] > 0 and not r["truncated_ctx"]]
    recs = [r for i, r in enumerate(recs) if i % args.num_shards == args.shard_id]
    print(f"[feat] shard {args.shard_id}/{args.num_shards}: {len(recs)} trajectories",
          flush=True)

    all_rows, all_feats, n_mismatch = [], [], 0
    meta_path = os.path.join(args.out_dir, f"meta.shard{args.shard_id}.jsonl")
    with open(meta_path, "w") as fmeta:
        for i, rec in enumerate(recs):
            try:
                rows, feats, mm = extract_one(
                    model, tokenizer, conv_start, rec, layer_ids, args.device)
            except Exception as exc:
                if not args.skip_anchor_failures:
                    raise
                print(f"[feat][skip] task={rec.get('task_idx')} "
                      f"k={rec.get('rollout_k')} error={type(exc).__name__}: {exc}",
                      flush=True)
                continue
            n_mismatch += mm
            all_feats.append(feats)
            for r in rows:
                fmeta.write(json.dumps(r) + "\n")
            all_rows.extend(rows)
            if (i + 1) % 50 == 0:
                print(f"[feat] {i + 1}/{len(recs)}, "
                      f"anchors={len(all_rows)}, token-mismatch rounds={n_mismatch}",
                      flush=True)

    if not all_feats:
        raise SystemExit("no features extracted")
    X = np.concatenate(all_feats, axis=0)
    np.savez(os.path.join(args.out_dir, f"feat.shard{args.shard_id}.npz"),
             X=X, layer_ids=np.array(layer_ids))
    print(f"[feat] done: X{X.shape}, total mismatch rounds {n_mismatch} -> {args.out_dir}",
          flush=True)

if __name__ == "__main__":
    main()
