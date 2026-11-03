"""Correctness checks against independent finite-code/tree enumeration."""
import itertools
import unittest
import numpy as np
from policy_osd import (Exchange, ModelBranchPolicy, PruneDecision,
                        conventional_basis, decode, prepare_basis)
from policy_osd.algebra import gf2_inverse, gf2_rank
from policy_osd.demo import hamming_generator
from policy_osd.refinement import legal_exchanges
from policy_osd.tree import children


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.generator=hamming_generator()
        self.messages=np.array(list(itertools.product((0,1),repeat=4)),dtype=np.uint8)
        self.codewords=self.messages@self.generator%2
        self.rng=np.random.default_rng(20261103)

    def test_tree_is_exact_without_duplicates(self):
        for k in range(1,9):
            for order in range(k+1):
                expected={a for w in range(order+1) for a in itertools.combinations(range(k),w)}
                stack=[()];seen=[]
                while stack:
                    node=stack.pop();seen.append(node)
                    stack.extend(children(node,k,order).values())
                self.assertEqual(len(seen),len(set(seen)))
                self.assertEqual(set(seen),expected)

    def test_gf2_inverse_and_reject_singular(self):
        matrix=np.array([[1,1,0],[0,1,1],[1,1,1]],dtype=np.uint8)
        np.testing.assert_array_equal(matrix@gf2_inverse(matrix)%2,np.eye(3,dtype=np.uint8))
        with self.assertRaises(ValueError):
            gf2_inverse(np.ones((3,3),dtype=np.uint8))

    def test_basis_is_greedy_reliability_order(self):
        received=self.rng.normal(size=7)
        expected=[]
        for col in np.argsort(-abs(received),kind='stable'):
            candidate=expected+[int(col)]
            if gf2_rank(self.generator[:,candidate])==len(candidate):
                expected=candidate
            if len(expected)==4:break
        self.assertEqual(conventional_basis(self.generator,received),tuple(expected))

    def test_noise_free_all_messages(self):
        for word in self.codewords:
            result=decode(self.generator,1.-2.*word.astype(float),order=0)
            np.testing.assert_array_equal(result.codeword,word)
            self.assertEqual(result.checked,1)

    def test_full_order_matches_exhaustive_ml_under_both_policies(self):
        alternative=ModelBranchPolicy(lambda observation:[0.,1.])
        for _ in range(32):
            received=self.rng.normal(size=7)
            distances=((received[None]-(1.-2.*self.codewords.astype(float)))**2).sum(1)
            expected=self.codewords[distances.argmin()]
            for policy in (None,alternative):
                result=decode(self.generator,received,order=4,branch_policy=policy,trace=True)
                np.testing.assert_array_equal(result.codeword,expected)
                self.assertEqual(result.checked,16)
                self.assertEqual(len(set(result.checked_supports)),16)
                self.assertEqual(result.reason,'space_exhausted')

    def test_refinement_preserves_code_and_rebuilds_model_input(self):
        received=self.rng.normal(size=7)
        initial=prepare_basis(self.generator,received)
        action=legal_exchanges(initial)[0]
        calls=[]
        def choose(observation):
            np.testing.assert_array_equal(observation.basis.systematic[:,:4],np.eye(4,dtype=np.uint8))
            self.assertNotEqual(set(observation.basis.positions),set(initial.positions))
            self.assertTrue(np.isfinite(observation.normalized_signed_distance).all())
            calls.append(observation)
            return [1.,0.]
        base=decode(self.generator,received,order=4)
        refined=decode(self.generator,received,order=4,
            mrb_policy=lambda view,legal,step:action if step==0 else None,
            branch_policy=ModelBranchPolicy(choose))
        np.testing.assert_array_equal(base.codeword,refined.codeword)
        self.assertEqual(refined.exchanges,(action,))
        self.assertTrue(calls)
        view=prepare_basis(self.generator,received,refined.basis)
        generated={tuple(view.encode_tep(tep)) for tep in self.messages}
        self.assertEqual(generated,{tuple(word) for word in self.codewords})

    def test_immediate_reversal_is_excluded(self):
        view=prepare_basis(self.generator,self.rng.normal(size=7))
        move=legal_exchanges(view)[0]
        positions=[move.incoming if i==move.outgoing else i for i in view.positions]
        child=prepare_basis(self.generator,view.received,positions)
        self.assertNotIn(Exchange(move.incoming,move.outgoing),legal_exchanges(child,view.positions))

    def test_stop_and_budget(self):
        received=self.rng.normal(size=7)
        result=decode(self.generator,received,order=4,stopping=lambda progress:True)
        self.assertEqual((result.checked,result.reason),(1,'stopping_rule'))
        capped=decode(self.generator,received,order=4,budget=3)
        self.assertEqual((capped.checked,capped.reason),(3,'budget_exhausted'))

    def test_candidate_pruning_keeps_descendants(self):
        received=self.rng.normal(size=7)
        result=decode(self.generator,received,order=4,trace=True,
            pruning=lambda basis,support,best:PruneDecision.SKIP_CANDIDATE if not support else PruneDecision.KEEP)
        self.assertEqual(result.checked,15)
        self.assertEqual(result.visited,16)
        self.assertNotIn((),result.checked_supports)
        self.assertIn((0,1,2,3),result.checked_supports)

    def test_empty_pruned_search_is_reported(self):
        result=decode(self.generator,self.rng.normal(size=7),order=4,
            pruning=lambda basis,support,best:PruneDecision.SKIP_SUBTREE)
        self.assertIsNone(result.codeword)
        self.assertEqual((result.checked,result.reason),(0,'no_candidate'))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):decode(self.generator,np.zeros(7),budget=0)
        with self.assertRaises(ValueError):decode(self.generator,np.zeros(7),order=5)
        with self.assertRaises(ValueError):prepare_basis(self.generator,np.zeros(6))
        with self.assertRaises(ValueError):prepare_basis(self.generator,np.zeros(7),[0,0,1,2])
        with self.assertRaises(ValueError):decode(self.generator,np.zeros(7),mrb_policy=lambda *args:Exchange(0,0))
        with self.assertRaises(ValueError):decode(self.generator,np.zeros(7),order=2,branch_policy=ModelBranchPolicy(lambda obs:[np.nan,0.]))

    def test_observations_are_read_only_and_have_no_reference_word(self):
        records=[]
        def model(observation):
            records.append(observation)
            self.assertFalse(hasattr(observation,'transmitted'))
            self.assertFalse(hasattr(observation,'reference'))
            with self.assertRaises(ValueError):observation.signed_distance[0]=0.
            return [0.,1.]
        result=decode(self.generator,self.rng.normal(size=7),order=4,branch_policy=ModelBranchPolicy(model))
        self.assertEqual(result.policy_calls,len(records))
        self.assertTrue(records)


if __name__=='__main__':
    unittest.main()
