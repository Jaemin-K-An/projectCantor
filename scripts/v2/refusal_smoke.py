"""GO/NO-GO (harness STOP CONDITION A precursor): does the model refuse at all?

If a model shows no refusal behaviour there is no refusal direction to steer
and the whole LLM phase is vacuous. Run this before anything else.
No completion text is printed -- only aggregate rates.
"""
import sys, time, torch
sys.path.insert(0, "llm/src")
from cantor_guard.models import load_model, chat_prompt
from cantor_guard.safety_eval import is_refusal, HARMFUL_SMOKE, BENIGN_SMOKE

for key in sys.argv[1:]:
    t0 = time.time()
    b = load_model(key)
    print(f"\n=== {key} ({b.repo}) layers={b.n_layers} d={b.d_model} "
          f"device={b.device} dtype={b.dtype} loaded {time.time()-t0:.1f}s")
    for name, prompts in (("HARMFUL", HARMFUL_SMOKE), ("BENIGN", BENIGN_SMOKE)):
        n = 0
        for p in prompts:
            ids = b.tokenizer(chat_prompt(b, p), return_tensors="pt").to(b.device)
            with torch.no_grad():
                out = b.model.generate(**ids, max_new_tokens=40, do_sample=False,
                                       pad_token_id=b.tokenizer.pad_token_id)
            comp = b.tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
            n += is_refusal(comp)
        print(f"  {name:8s} refusal rate = {n}/{len(prompts)} = {n/len(prompts):.2f}")
    del b
