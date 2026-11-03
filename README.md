# MCTS-Decoding: Initial Framework Release

This repository accompanies our work on policy-guided decoding of short linear
block codes, including our ICC 2026 paper. This first release provides a small,
runnable receiver framework and extension interfaces.

**Release scope:** the learned network implementations, pretrained weights,
MCTS training/label-generation pipelines, training data, and paper-scale
experiments are not included. This release demonstrates the software interfaces;
it does **not** reproduce the paper's BLER, checked-TEP, or timing curves.

## Included in this release

| Component | Available |
| --- | --- |
| GF(2) linear algebra and reliability-based independent-basis selection | Yes |
| Systematic re-encoding and original-coordinate reconstruction | Yes |
| Extend/adjacent TEP tree and policy-guided DFS | Yes |
| Candidate-check budgets and best-candidate bookkeeping | Yes |
| Stopping and candidate/subtree pruning interfaces | Yes |
| Rank-preserving MRB exchange and STOP interface | Yes |
| Adapter for an external model's two action scores | Interface only |
| Learned GNN/MRB architectures and model checkpoints | Not included |
| MCTS teacher, training, and full experimental pipelines | Not included |

The built-in executable policy is a **non-neural reliability heuristic**. It is
not a trained model and is not the GNN used in the paper. The framework has no
PyTorch, GPU, model-download, or external-data dependency.

## Quick start

Requires Python 3.10 or later and NumPy.

```bash
git clone https://github.com/1014Tea/MCTS-Decoding.git
cd MCTS-Decoding
python -m pip install -e .
python -m policy_osd --trials 100 --snr-db 2 --order 2
```

The example transmits uniformly random messages using a systematic Hamming
`(7,4)` code over BPSK/AWGN. The fixed random seed makes the small smoke test
repeatable. SNR means `10*log10(1/noise_variance)` for unit-amplitude BPSK; it is
not an Eb/N0 conversion. Printed BLER values are example results only.

No early stopping or pruning is enabled by default. DFS exhausts the
order-limited TEP set unless a candidate-check budget is supplied:

```bash
python -m policy_osd --trials 100 --order 2 --budget 5
python -m unittest discover -s tests -v
python examples/hook_demo.py
```

## Receiver flow

```text
received samples + generator matrix
        |
reliability ordering + independent basis + systematization
        |
optional MRB-policy hook: legal exchange or STOP, bounded depth
        |
rebuild the selected basis and its coordinate-dependent objects
        |
TEP tree + policy-guided depth-first traversal
        |
candidate re-encoding -> distance -> best candidate -> stopping hook
        |
decoded codeword in the original code-position order
```

```python
from policy_osd import decode

result = decode(G, received, order=2, budget=100)
print(result.codeword, result.checked, result.reason)
```

The receiver API accepts no transmitted/reference codeword. In the demo, the
transmitted word is used only by the surrounding evaluator to count errors.

## Model and MRB interfaces

`ModelBranchPolicy(predict)` accepts a user-supplied function returning two
finite scores in `[extend, adjacent]` order. The framework masks unavailable
actions and queries the function only at nodes with two legal children.

```python
from policy_osd import ModelBranchPolicy, decode

def predict(observation):
    # Attach your own model here. No trained network is bundled.
    # Current inputs include:
    # observation.basis.systematic: k-by-n systematic generator
    # observation.tep: current k-bit TEP
    # observation.signed_distance: (r' - x')**2 * x'
    # observation.normalized_signed_distance: scaled per current candidate
    return [0.0, 1.0]  # Shape-only example; these are not learned scores.

result = decode(G, received, order=2,
                branch_policy=ModelBranchPolicy(predict))
```

An MRB-policy hook receives the current `BasisView`, the legal exchanges, and
the exchange step. It returns an `Exchange(outgoing, incoming)` or `None`
for STOP. The framework maintains full rank, forbids an immediate reversal,
and rebuilds the graph inputs after a selected exchange. The hook contains
no learned MRB selector or MCTS implementation.

See [the executable hook example](examples/hook_demo.py).
It uses one offline-selected Hamming-code sample (seed 2331, noise standard
deviation 0.8). Both runs use the same fixed branch scores; the refinement hook
makes the first legal exchange and then stops, without access to the target.
The target codeword is visited fifth with the conventional basis and first
after the exchange. Both runs still check all 11 order-two TEPs. The printed
target position is a one-based offline statistic, not an online stopping rule
or evidence of average performance gains.

## Stopping, pruning, and counting

- `stopping(progress)` runs after each evaluated candidate and returns a Boolean.
  It can inspect the current and best candidates, their distances, and counts.
  No PB-OSD confidence formula or calibrated threshold is bundled in this release.
- `pruning(basis, support, best)` returns `PruneDecision.KEEP`,
  `SKIP_CANDIDATE`, or `SKIP_SUBTREE`. Skipping only a candidate keeps its
  descendants reachable. A subtree should be discarded only when the supplied
  rule justifies removing all its descendants.
- Rules that assume a particular enumeration order must be adapted accordingly.
  The callback interface itself does not validate an external rule's error-rate
  guarantees.
- `checked` counts candidate distance evaluations; `visited` counts popped
  tree states, including pruned ones. `policy_calls` counts branching decisions.
  A candidate skipped by a pruning callback may still need re-encoding to form
  an external policy's features; it is not counted as a distance evaluation.
- The returned best codeword is in the **original** coordinate order. If every
  candidate is pruned, `codeword` is `None` and `reason` is `no_candidate`.

For a fixed basis, exhaustive traversal with no stopping/pruning examines the
same order-limited candidate set regardless of branch priorities. A different
basis, a finite budget, early stopping, or pruning can change the output.

## Layout

```text
policy_osd/
  algebra.py       GF(2), basis selection, systematic representation
  tree.py          deterministic TEP-tree operations
  interfaces.py    policy, refinement, stopping, and pruning contracts
  refinement.py    legal exchanges and bounded refinement
  decoder.py       receiver orchestration and DFS
  demo.py          standalone Hamming-code example
examples/
  hook_demo.py     connect external policy and receiver hooks
tests/
  test_framework.py
```

## License

The code in this framework release is distributed under the [MIT License](LICENSE).
No unpublished manuscripts, private datasets, or pretrained model weights are
distributed in this repository.
