"""Train the mini-LLaVA projector (stage 1), then projector + LoRA adapters on the LM (stage 2).

    python 09b_mini_llava/train.py --stage projector
    python 09b_mini_llava/train.py --stage lora --init-from 09b_mini_llava/runs/projector_s0
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import shapes  # noqa: E402
from common.utils import Logger, get_device, make_run_dir, save_json, seed_everything  # noqa: E402
from model import Projector, add_lora, build_inputs, load_language_model  # noqa: E402
from objective import answer_loss  # noqa: E402
from sample import cached_features, evaluate, qa_items  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="projector", choices=["projector", "lora"])
    p.add_argument("--init-from", default=None, help="stage lora: run folder with the trained projector")
    p.add_argument("--train-images", type=int, default=20000)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    if args.stage == "lora" and args.init_from is None:
        p.error("--stage lora needs --init-from")
    if args.name is None:
        args.name = f"{args.stage}_s{args.seed}"
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    run_dir = make_run_dir(__file__, args.name)
    save_json(vars(args), run_dir / "args.json")

    train_feats = cached_features("train", device, n=args.train_images).to(device)
    test_feats = cached_features("test", device).to(device)
    train_scenes = shapes.load("train")["scenes"][:len(train_feats)]
    test_scenes = shapes.load("test")["scenes"][:1000]          # quick evaluation during training

    lm, tok = load_language_model(device)
    projector = Projector().to(device)
    params = list(projector.parameters())
    if args.stage == "lora":
        projector.load_state_dict(torch.load(Path(args.init_from) / "ckpt.pt", map_location=device)["projector"])
        lm = add_lora(lm)
        params += [p for p in lm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr if args.stage == "projector" else args.lr / 5, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=args.steps,
                                                pct_start=0.05)
    logger = Logger(run_dir)
    items = qa_items(train_scenes, seed=args.seed)
    rng = random.Random(args.seed)

    for step in range(1, args.steps + 1):
        lm.train() if args.stage == "lora" else lm.eval()
        projector.train()
        batch = rng.sample(items, args.batch_size)
        idx = torch.tensor([i for i, _, _, _ in batch], device=device)
        inputs, mask, targets, _ = build_inputs(lm, tok, projector(train_feats[idx].float()),
                                                [q for _, q, _, _ in batch], [a for _, _, a, _ in batch])
        logits = lm(inputs_embeds=inputs, attention_mask=mask).logits
        loss = answer_loss(logits[:, :-1], targets[:, :-1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            lm.eval()
            projector.eval()
            logger.log(step, loss=loss.item(), **evaluate(lm, tok, projector, test_feats, test_scenes))

    lora = {k: v for k, v in lm.state_dict().items() if "lora_" in k} if args.stage == "lora" else None
    torch.save({"projector": projector.state_dict(), "lora": lora, "args": vars(args)}, run_dir / "ckpt.pt")
    n_trained = sum(p.numel() for p in params)
    save_json({"trained_params": n_trained}, run_dir / "params.json")


if __name__ == "__main__":
    main()
