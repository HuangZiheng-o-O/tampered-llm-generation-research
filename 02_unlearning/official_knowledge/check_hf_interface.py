"""Exercise official HF evaluation with a tiny random CPU Llama, never a trainer."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--suite-root", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.upstream / "src"))
    from evals.lm_eval import LMEvalEvaluator
    from lm_eval.api.instance import Instance
    torch.set_num_threads(2)
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=32000, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096,
                         bos_token_id=1, eos_token_id=2, pad_token_id=2)
    config._name_or_path = str(args.tokenizer)
    toy = LlamaForCausalLM(config).eval()
    # Use the official wrapper, including its own tokenizer loading path.
    evaluator = LMEvalEvaluator(OmegaConf.create({"tasks": [], "include_path": str(args.suite_root / "tasks"), "simple_evaluate_args": {}}))
    wrapped = evaluator.prepare_model(toy)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    doc = json.loads((args.suite_root / "data/biographies/eval-development.jsonl").read_text().splitlines()[0])
    assert wrapped.tok_encode(doc["prompt"]) == tokenizer.encode(doc["prompt"], add_special_tokens=True)
    generation = Instance(request_type="generate_until", doc=doc, arguments=(doc["prompt"], {"until": ["\n"], "max_gen_toks": 2, "do_sample": False}), idx=0)
    likelihood = Instance(request_type="loglikelihood", doc=doc, arguments=(doc["prompt"], " " + doc["answers"][0]), idx=0)
    with torch.inference_mode():
        generated = wrapped.generate_until([generation], disable_tqdm=True)
        likelihoods = wrapped.loglikelihood([likelihood], disable_tqdm=True)
    assert len(generated) == len(likelihoods) == 1 and math.isfinite(likelihoods[0][0])
    report = {"proof_kind": "tiny_random_cpu_model_interface_only", "device": str(toy.device),
              "parameters": sum(p.numel() for p in toy.parameters()), "pretrained_weights_loaded": False,
              "optimizer_steps": 0, "official_hf_generation": "passed", "official_hf_loglikelihood": "passed",
              "actual_tokenizer_and_bos_alignment": "passed"}
    (args.suite_root / "hf-interface-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
