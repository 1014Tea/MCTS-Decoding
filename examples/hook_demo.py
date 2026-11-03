"""Connect replaceable interfaces, without any trained model or checkpoint."""
from dataclasses import asdict
import json
import numpy as np
from policy_osd import ModelBranchPolicy, PruneDecision, decode
from policy_osd.demo import hamming_generator


def example_scores(observation):
    # Replace with a model call returning [extend_score, adjacent_score].
    # These fixed scores demonstrate the API; they are not learned weights.
    assert observation.basis.systematic.shape == (4,7)
    assert np.array_equal(observation.basis.systematic[:,:4],np.eye(4,dtype=np.uint8))
    return [0.,1.]


def example_refinement(basis,legal,step):
    # Deliberately simple interface example, not a performance optimization.
    return legal[0] if step==0 and legal else None


def example_stop(progress):
    # Exact observation/codeword agreement; not the PB-OSD approximation.
    return progress.best.squared_distance < 1e-12


def example_prune(basis,support,best):
    # Attach a valid candidate or subtree discarding rule here.
    return PruneDecision.KEEP


def main():
    generator=hamming_generator()
    message=np.array([1,0,1,0],dtype=np.uint8)
    transmitted=message@generator%2
    received=1.-2.*transmitted.astype(float)+.2*np.random.default_rng(7).normal(size=7)
    result=decode(generator,received,order=2,
        branch_policy=ModelBranchPolicy(example_scores),mrb_policy=example_refinement,
        stopping=example_stop,pruning=example_prune)
    print(json.dumps(dict(policy='fixed-score interface demonstration, not a trained model',
        selected_basis=result.basis,exchanges=[asdict(a) for a in result.exchanges],
        checked=result.checked,policy_calls=result.policy_calls,
        decoded=result.codeword.tolist() if result.codeword is not None else None,
        correct=bool(np.array_equal(result.codeword,transmitted))),indent=2))


if __name__=='__main__':
    main()
