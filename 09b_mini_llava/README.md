# 09b Mini-LLaVA

Teach a small frozen language model to answer questions about shape scenes by feeding it image features
as if they were words. Only a small projector (and later low-rank adapters) is trained.

Uses pretrained [SigLIP-base](https://huggingface.co/google/siglip-base-patch16-224) (vision) and
[SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) (language), both frozen.

## Math

**A language model is an autoregressive model** (as in 04 and 05, over words):
$p(w_1, \dots, w_n) = \prod_t p(w_t \mid w_{<t})$. Each word id $w$ enters as a vector $E_w \in \mathbb R^{576}$.

**Visual tokens.** SigLIP turns the image into 196 patch vectors (768-d); 2 x 2 averaging leaves 49.
A projector $P$ (MLP) maps each to $\mathbb R^{576}$, the LM's word space. These 49 vectors are placed before the question:

$$p(\text{answer} \mid x, q) = \prod_t p\left(a_t \mid P(\mathrm{SigLIP}(x))_{1..49},\ E_{q},\ a_{<t}\right)$$

Example: for "What color is the shape on the left?" the LM sees 49 visual vectors, then the question tokens, and should put
most probability on the token " red".

**Loss.** Cross-entropy on the answer tokens only:
$L = -\sum_{t \in \text{answer}} \log p(a_t \mid \cdot)$.

**LoRA.** Keep a weight matrix $W \in \mathbb R^{d_{out} \times d_{in}}$ frozen and learn a low-rank change:

$$W' = W + \frac{\alpha}{r} B A, \qquad A \in \mathbb R^{r \times d_{in}},\ B \in \mathbb R^{d_{out} \times r},\ B = 0 \text{ at the start}$$

Example: a 576 x 576 query projection has 331,776 weights; with $r = 8$ the adapter has $8 \times 576 + 576 \times 8 = 9{,}216$ (2.8 %).

**Control.** Pair each question with another image's features. If accuracy drops to the level of guessing,
the answers really come from the image.

## Code map

| Math | Code |
|---|---|
| SigLIP features, 2 x 2 pooling | `model.py: image_features` |
| $P$: 768 -> 576 -> 576 | `model.py: Projector` |
| [visual tokens][question][answer], targets | `model.py: build_inputs` |
| $W + \frac{\alpha}{r} BA$ on query and value projections | `model.py: add_lora` |
| answer-only cross-entropy | `objective.py: answer_loss` |
| greedy one-word answer, accuracy per type, shuffled control | `sample.py: answer`, `evaluate` |

## Setup

| | |
|---|---|
| Data | shape scenes (32 x 32, upsampled to 224 for SigLIP); 20k training images, 5k test images |
| Questions | count; shape, color and size at each position; "Is there a <color> <shape>?" (yes / no) |
| Stage 1 | projector only (0.78 M parameters); AdamW, lr 1e-3 (one-cycle), batch 64, 3000 steps |
| Stage 2 | projector + LoRA ($r = 8$, $\alpha = 16$, query and value projections; 1.24 M parameters); lr 2e-4, 3000 steps |
| Metric | exact-match accuracy of the first answer token, per question type, on all questions of the 5k test scenes |

## Run

```
bash 09b_mini_llava/run.sh          # features, stage 1, stage 2, evaluation
python 09b_mini_llava/figures.py    # figures and table
```

## Checklist

- [x] Stage 1: projector only, language model frozen
- [x] Stage 2: projector + LoRA
- [x] Control: questions paired with other images

## Results

Accuracy on all questions of the 5000 test scenes (about 40k questions).

| Question type | Projector only (0.78 M trained) | Projector + LoRA (1.24 M trained) | Other image's features (LoRA model) |
|---|---|---|---|
| count | 0.999 | 1.000 | 0.319 |
| shape | 0.934 | 0.999 | 0.331 |
| color | 0.797 | 0.999 | 0.260 |
| size | 0.987 | 0.998 | 0.500 |
| exists (yes / no) | 0.914 | 0.932 | 0.494 |
| all | 0.919 | 0.991 | 0.374 |

- **A projector alone gets most of the way.** With the language model untouched, 0.78 M trained parameters answer
  92 % of questions. Count and size are near perfect from the start.
- **LoRA fixes color and shape.** Adding 0.46 M adapter weights inside the LM lifts color from 80 % to 99.9 % and
  shape from 93 % to 99.9 % within 1000 steps. The frozen LM can read the visual tokens, but tying "the shape on the
  left" to the right visual tokens needed a small change inside attention.
- **"Is there a red square?" stays hardest** (93 %). It needs color and shape bound together and a check over every
  object, the same binding problem as in 09a.
- **The answers come from the image.** With another image's features, accuracy falls to the level of guessing:
  1/4 for color, 1/3 for shape and count, 1/2 for size and yes / no.
