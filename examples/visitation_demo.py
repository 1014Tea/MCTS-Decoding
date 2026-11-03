"""Selected examples of policy ordering, MRB exchange, and their combination.

All receiver rules use only received samples and the code. The transmitted
codeword is used afterward to measure first visitation, never to choose an
action or stop decoding. These illustrations do not estimate average gains.
"""
import argparse
from dataclasses import asdict
import itertools
import json

import numpy as np

from policy_osd import ModelBranchPolicy, ReliabilityPolicy, decode, prepare_basis
from policy_osd.tree import dense


METHOD_LABELS = {
    "baseline": "Fixed extend-first DFS",
    "policy_only": "Reliability policy only",
    "mrb_only": "MRB exchange only",
    "combined": "MRB exchange + policy",
}
CASE_SEEDS = {"policy": 34, "combined": 412}


def rm_1_4_generator():
    """Evaluate the constant and four coordinate functions on all binary points."""
    points = np.array(list(itertools.product((0, 1), repeat=4)), dtype=np.uint8)
    return np.vstack([np.ones(16, dtype=np.uint8), points.T])


def first_legal_exchange(basis, legal, step):
    """One deterministic legal exchange; no target-dependent scoring."""
    return legal[0] if step == 0 and legal else None


def summarize(result, generator, received, target):
    """Reconstruct the checked candidates for offline reporting only."""
    view = prepare_basis(generator, received, result.basis)
    trace = []
    target_position = None
    for position, support in enumerate(result.checked_supports, start=1):
        word = view.encode_tep(dense(support, view.k))
        matches = bool(np.array_equal(word, target))
        trace.append(dict(position=position, support=list(support),
                          codeword="".join(map(str, word.tolist())), is_target=matches))
        if matches:
            target_position = position
    target_tep = view.hard_information ^ target[list(view.positions)]
    return dict(
        target_first_checked_position=target_position,
        checked=result.checked,
        selected_basis=list(result.basis),
        exchanges=[asdict(action) for action in result.exchanges],
        target_basis_error_positions=[view.positions[i] for i in np.flatnonzero(target_tep)],
        decoded=result.codeword.tolist() if result.codeword is not None else None,
        correct=bool(np.array_equal(result.codeword, target)),
        trace=trace,
    )


def run_case(name):
    generator = rm_1_4_generator()
    message = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
    target = message @ generator % 2
    seed, noise_std = CASE_SEEDS[name], 1.3
    received = (1. - 2. * target.astype(float)
                + noise_std * np.random.default_rng(seed).normal(size=16))
    fixed = ModelBranchPolicy(lambda observation: [1., 0.])
    guided = ReliabilityPolicy(order=2)
    settings = {
        "baseline": dict(branch_policy=fixed),
        "policy_only": dict(branch_policy=guided),
        "mrb_only": dict(branch_policy=fixed, mrb_policy=first_legal_exchange),
        "combined": dict(branch_policy=guided, mrb_policy=first_legal_exchange),
    }
    methods = {}
    for method, options in settings.items():
        result = decode(generator, received, order=2, trace=True, **options)
        methods[method] = summarize(result, generator, received, target)
    return dict(name=name, seed=seed, noise_std=noise_std, received=received.tolist(),
                target="".join(map(str, target.tolist())), methods=methods)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "policy", "combined"), default="all")
    parser.add_argument("--json", action="store_true", help="Include every checked candidate")
    args = parser.parse_args()
    names = CASE_SEEDS if args.case == "all" else (args.case,)
    report = dict(
        code="Reed-Muller RM(1,4), (n,k,d)=(16,5,8)",
        order=2,
        baseline="fixed extend-first DFS; no early stopping or pruning",
        policy="non-neural policy prioritizing the child TEP with lower weighted reliability cost",
        refinement="first legal exchange once, then STOP; same rule for both refined runs",
        measurement="one-based target position, measured offline after complete decoding",
        scope="offline-selected examples; no learned networks or average-performance claim",
        cases=[run_case(name) for name in names],
    )
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(report["code"] + "; order 2; target 1010010110100101")
    print("Policy: non-neural reliability rule; baseline: fixed extend-first DFS.")
    print("First target position is measured offline. All order-two TEPs are checked.")
    for case in report["cases"]:
        print(f"\n{case['name']} example: seed={case['seed']}, noise_std={case['noise_std']}")
        print(f"{'Method':<29} {'Target position':>15} {'Checked TEPs':>13}")
        for method, summary in case["methods"].items():
            position = summary["target_first_checked_position"]
            print(f"{METHOD_LABELS[method]:<29} {str(position):>15} {summary['checked']:>13}")
    print("\nThese are selected illustrations. Use --json for bases, exchanges, and full traces.")


if __name__ == "__main__":
    main()
