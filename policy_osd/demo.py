"""Small CPU smoke demonstration, not a reproduction of paper BLER curves."""
from __future__ import annotations
import argparse
import json
import numpy as np
from .decoder import decode


def hamming_generator():
    return np.array([[1,0,0,0,1,1,0],
                     [0,1,0,0,1,0,1],
                     [0,0,1,0,0,1,1],
                     [0,0,0,1,1,1,1]],dtype=np.uint8)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=int,default=100)
    parser.add_argument('--snr-db',type=float,default=2.)
    parser.add_argument('--order',type=int,default=2)
    parser.add_argument('--seed',type=int,default=20261103)
    parser.add_argument('--budget',type=int)
    args=parser.parse_args()
    if args.trials<1 or not np.isfinite(args.snr_db):
        parser.error('trials must be positive and snr-db must be finite')
    rng=np.random.default_rng(args.seed);generator=hamming_generator()
    errors=checked=calls=0
    for _ in range(args.trials):
        message=rng.integers(0,2,4,dtype=np.uint8)
        codeword=message@generator%2
        received=1.-2.*codeword.astype(float)+10.**(-args.snr_db/20.)*rng.normal(size=7)
        result=decode(generator,received,order=args.order,budget=args.budget)
        errors+=int(not np.array_equal(result.codeword,codeword))
        checked+=result.checked;calls+=result.policy_calls
    print(json.dumps(dict(release='framework-only',code=[7,4],trials=args.trials,
        snr_db=args.snr_db,snr_definition='10*log10(1/noise_variance), BPSK amplitude1',
        order=args.order,seed=args.seed,policy='non-neural reliability fallback',
        pretrained_weights_used=False,stopping='budget or exhaustive order-limited list',
        block_errors=errors,bler=errors/args.trials,mean_checked=checked/args.trials,
        mean_policy_calls=calls/args.trials),indent=2))


if __name__=='__main__':
    main()
