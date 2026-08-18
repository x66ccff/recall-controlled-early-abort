#!/usr/bin/env python

import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

def extract_action(text):

    m = re.findall(r"Action:\s*(.*?)(?=\n|$)", text, re.DOTALL)
    if len(m) != 1:
        return f"<PARSE_FAIL:{len(m)} matches>"
    a = re.sub(r"[^A-Za-z0-9, ]+", "", m[-1])
    return " ".join(a.split()).strip()

FIXED_SYSTEM = os.environ.get(
    "AGENT_EARLY_ABORT_FIXED_SYSTEM", "You are a helpful assistant.")
FIXED_DATE = "26 Jul 2024"
ENABLE_THINKING_ENV = os.environ.get("AGENT_EARLY_ABORT_ENABLE_THINKING")

def chat_template_kwargs():
    kwargs = {"date_string": FIXED_DATE}
    if ENABLE_THINKING_ENV is not None:
        kwargs["enable_thinking"] = ENABLE_THINKING_ENV.lower() not in {
            "0", "false", "no", "off"}
    return kwargs

def build_prompt(tokenizer, conversation):

    role_map = {"human": "user", "gpt": "assistant"}
    msgs = [{"role": "system", "content": FIXED_SYSTEM}]
    for m in conversation:
        role = role_map[m["from"]]
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n" + m["value"]
        else:
            msgs.append({"role": role, "content": m["value"]})
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        **chat_template_kwargs())

def load_done_keys(out_path):

    done = set()
    for path in glob.glob(out_path + "*"):
        try:
            with open(path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        done.add((int(r["task_idx"]), int(r["rollout_k"])))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except OSError:
            continue
    return done

def worker_main(rank, args, items):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    if args.env == "textcraft":
        from agentenv.envs import TextCraftEnvClient as EnvClient
    elif args.env == "webshop":
        from agentenv.envs.webshop import WebshopEnvClient as EnvClient
    elif args.env == "alfworld":
        from agentenv.envs.alfworld import AlfWorldEnvClient as EnvClient
    else:
        raise ValueError(f"unsupported env: {args.env}")

    def log(msg):
        print(f"[worker{rank}] {msg}", flush=True)

    if not items:
        log("no pending items; exiting")
        return

    log(f"loading vLLM engine: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        seed=args.seed + rank,
        disable_log_stats=True,
    )

    servers = [s.strip().rstrip("/") for s in args.server_base.split(",")]
    base = servers[rank % len(servers)]

    def new_client():
        last_err = None
        for _ in range(3):
            try:
                return EnvClient(
                    env_server_base=base,
                    data_len=max(args.n_tasks, 1),
                    timeout=args.http_timeout,
                )
            except Exception as e:
                last_err = e
                time.sleep(3)
        raise RuntimeError(f"client creation failed after 3 attempts: {last_err}")

    pending = list(items)
    n_slots = min(args.concurrency, len(pending))
    clients = [new_client() for _ in range(n_slots)]
    pool = ThreadPoolExecutor(max_workers=n_slots)

    out_path = f"{args.out}.rank{rank}.jsonl"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fout = open(out_path, "a", buffering=1)
    n_done_total = 0

    slots = [None] * n_slots

    def write_reset_error(task_idx, k, error, t0):
        nonlocal n_done_total
        rec = {
            "task_idx": task_idx,
            "rollout_k": k,
            "init_obs": "",
            "n_rounds": 0,
            "done": False,
            "final_reward": 0.0,
            "success": 0,
            "hit_max_rounds": False,
            "truncated_ctx": False,
            "reset_error": str(error),
            "wall_time": round(time.time() - t0, 1),
            "rounds": [],
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_done_total += 1
        log(f"reset failed task={task_idx} k={k}: {error}")

    def start_slot(slot_id):
        while pending:
            task_idx, k = pending.pop(0)
            c = clients[slot_id]
            t0 = time.time()
            try:
                c.reset(task_idx)
                init_obs = c.observe()
            except Exception as e:
                write_reset_error(task_idx, k, e, t0)
                continue
            conv = [dict(m) for m in c.conversation_start]
            conv.append({"from": "human", "value": init_obs})
            return {
                "client": c, "task_idx": task_idx, "k": k,
                "conv": conv, "rounds": [], "t0": t0,
                "init_obs": init_obs, "final_reward": 0.0, "done": False,
                "truncated_ctx": False,
            }
        return None

    def finish(slot_id):
        nonlocal n_done_total
        s = slots[slot_id]
        rec = {
            "task_idx": s["task_idx"],
            "rollout_k": s["k"],
            "init_obs": s["init_obs"][:1500],
            "n_rounds": len(s["rounds"]),
            "done": s["done"],
            "final_reward": s["final_reward"],
            "success": int(s["done"] and s["final_reward"] > 0),
            "hit_max_rounds": len(s["rounds"]) >= args.max_rounds and not s["done"],
            "truncated_ctx": s["truncated_ctx"],
            "wall_time": round(time.time() - s["t0"], 1),
            "rounds": s["rounds"],
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_done_total += 1
        if n_done_total % 10 == 0:
            log(f"completed {n_done_total}, pending={len(pending)}")
        slots[slot_id] = start_slot(slot_id)

    for i in range(n_slots):
        slots[i] = start_slot(i)

    while any(s is not None for s in slots):
        batch_ids, prompts, sps = [], [], []
        for i, s in enumerate(slots):
            if s is None:
                continue

            if time.time() - s["t0"] > args.rollout_timeout:
                finish(i)
                continue
            prompt = build_prompt(tokenizer, s["conv"])
            n_tok = len(tokenizer(prompt).input_ids)
            if n_tok + args.max_new_tokens > args.max_model_len:
                s["truncated_ctx"] = True
                finish(i)
                continue

            seed = (args.seed * 1000003
                    + s["task_idx"] * 1009 + s["k"] * 101 + len(s["rounds"]))
            batch_ids.append(i)
            prompts.append(prompt)
            sps.append(SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
                logprobs=1,
                seed=seed,
            ))

        if not prompts:
            continue

        outs = llm.generate(prompts, sps, use_tqdm=False)

        def env_step(pair):
            slot_id, out = pair
            s = slots[slot_id]
            comp = out.outputs[0]
            text = comp.text
            lps = [d[t].logprob
                   for t, d in zip(comp.token_ids, comp.logprobs)]
            step_out = s["client"].step(text)
            return slot_id, text, lps, step_out

        for slot_id, text, lps, step_out in pool.map(
                env_step, zip(batch_ids, outs)):
            s = slots[slot_id]
            prompt_state = step_out.state or ""
            if args.env == "alfworld":

                prompt_state = s["client"].observe()
            s["rounds"].append({
                "round": len(s["rounds"]),
                "gen_text": text,
                "sent_action": extract_action(text),
                "token_logprobs": [round(x, 5) for x in lps],
                "n_gen_tokens": len(lps),
                "reward": float(step_out.reward),
                "done": bool(step_out.done),
                "obs_text": prompt_state[:500],
                "obs_len": len(prompt_state),
            })
            s["final_reward"] = float(step_out.reward)
            s["done"] = bool(step_out.done)
            s["conv"].append({"from": "gpt", "value": text})
            s["conv"].append({"from": "human", "value": prompt_state})
            if s["done"] or len(s["rounds"]) >= args.max_rounds:
                finish(slot_id)

    fout.close()
    log(f"all done, wrote {n_done_total} records -> {out_path}")

def merge_shards(out_path):

    records = {}
    sources = sorted(glob.glob(out_path + ".rank*.jsonl"))
    if os.path.exists(out_path):
        sources = [out_path] + sources
    for path in sources:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    records[(int(r["task_idx"]), int(r["rollout_k"]))] = line.rstrip("\n")
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
    with open(out_path, "w") as f:
        for key in sorted(records):
            f.write(records[key] + "\n")
    print(f"[merge] {len(records)} records -> {out_path}", flush=True)

def parse_task_ids(text):
    ids = []
    for part in re.split(r"[\s,]+", text.strip()):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return sorted(dict.fromkeys(ids))

def selected_task_ids(args):
    ids = list(range(args.n_tasks))
    if args.task_ids_file:
        with open(args.task_ids_file) as f:
            ids = parse_task_ids(f.read())
    if args.task_ids:
        ids = parse_task_ids(args.task_ids)
    return ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_base", required=True,
                    help="environment server address; comma-separated values distribute load")
    ap.add_argument("--env", choices=("textcraft", "webshop", "alfworld"), default="textcraft")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_tasks", type=int, default=10)
    ap.add_argument("--task_ids", "--task-ids", default="",
                    help="comma/space separated task ids or ranges, e.g. '0,3,8-12'")
    ap.add_argument("--task_ids_file", "--task-ids-file", default="",
                    help="file containing task ids or ranges; overrides --n_tasks")
    ap.add_argument("--k_rollouts", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_rounds", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--http_timeout", type=int, default=300)
    ap.add_argument("--rollout_timeout", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--n_gpus", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=128,
                    help="concurrent rollout slots per GPU")
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    args = ap.parse_args()

    done = load_done_keys(args.out) if args.resume else set()
    task_ids = selected_task_ids(args)
    items = [(t, k)
             for t in task_ids
             for k in range(args.k_rollouts)
             if (t, k) not in done]
    total = len(task_ids) * args.k_rollouts
    print(f"[rollout] total {total}, completed {total - len(items)}, "
          f"pending this run {len(items)}", flush=True)
    if not items:
        merge_shards(args.out)
        return

    n_workers = min(args.n_gpus, len(items))
    chunks = [items[r::n_workers] for r in range(n_workers)]

    mp.set_start_method("spawn", force=True)
    procs = []
    for rank in range(n_workers):
        p = mp.Process(target=worker_main, args=(rank, args, chunks[rank]))
        p.start()
        procs.append(p)
    failed = 0
    for p in procs:
        p.join()
        failed += int(p.exitcode != 0)
    if failed:
        print(f"[rollout] warning: {failed} workers exited nonzero; "
              f"rerun the command to resume", flush=True)

    merge_shards(args.out)

if __name__ == "__main__":
    main()
