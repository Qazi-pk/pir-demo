"""
M_11 Inertia-Tensor-Aware Library Test

The previous analysis showed:
  - Singles + Pairs + cos(2q):     R^2 = 0.852
  - + Triple products:              R^2 = 0.886  (only +3.4%, not enough)

The remaining ~12% residual is hypothesized to come from off-diagonal
inertia tensor components (ixy, ixz, iyz) in the Franka URDF. When these
non-zero off-diagonal terms are rotated through joint angles in the inertia
projection M_ii = sum_k J_omega^T R_k I_k R_k^T J_omega, they produce
sin(2q) and sin(q)*cos(q) terms that are NOT in the previous grammar.

Recall: sin(q)*cos(q) = sin(2q)/2, so adding sin(2q) is equivalent.

This script tests whether including sin(2q) atoms (and their products with
single trig terms) closes the R^2 gap. If yes, the missing structure is
identified and the grammar extension is principled. If no, we have to look
at PyBullet's specific mass-matrix implementation.

Library content:
  - Constant
  - Singles cos(q_j), sin(q_j) for j=2..7
  - Double-angle cos(2q_j), sin(2q_j) for j=2..7   <-- sin(2q) is the new addition
  - Pairwise products cos/sin(q_j) * cos/sin(q_k) for j<k
  - sin(2q_j) * cos(q_k) and sin(2q_j) * sin(q_k) for j != k   <-- new cross terms
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from itertools import combinations
import json
from pathlib import Path


def build_library_with_sin2q(df, joint_indices=range(2, 8)):
    """Build feature library including sin(2q) terms and their cross products."""
    joints = list(joint_indices)
    n_samples = len(df)
    
    feature_names = ['const']
    features = [np.ones(n_samples)]
    
    # Cache trig values
    cos_q = {j: np.cos(df[f'q{j}'].values) for j in joints}
    sin_q = {j: np.sin(df[f'q{j}'].values) for j in joints}
    cos_2q = {j: np.cos(2 * df[f'q{j}'].values) for j in joints}
    sin_2q = {j: np.sin(2 * df[f'q{j}'].values) for j in joints}
    
    # Singles
    for j in joints:
        feature_names += [f'cos(q{j})', f'sin(q{j})']
        features += [cos_q[j], sin_q[j]]
    
    # Double-angle terms (BOTH cos(2q) and sin(2q) - the new addition)
    for j in joints:
        feature_names += [f'cos(2q{j})', f'sin(2q{j})']
        features += [cos_2q[j], sin_2q[j]]
    
    # Pairwise products of singles (distinct joints)
    for j, k in combinations(joints, 2):
        for fj_name, fj_val in [('cos', cos_q[j]), ('sin', sin_q[j])]:
            for fk_name, fk_val in [('cos', cos_q[k]), ('sin', sin_q[k])]:
                feature_names.append(f'{fj_name}(q{j}){fk_name}(q{k})')
                features.append(fj_val * fk_val)
    
    # NEW: sin(2q_j) crossed with cos/sin(q_k) for distinct j, k
    # These arise from rotating off-diagonal inertia tensor components.
    for j in joints:
        for k in joints:
            if j == k:
                continue
            feature_names.append(f'sin(2q{j})cos(q{k})')
            features.append(sin_2q[j] * cos_q[k])
            feature_names.append(f'sin(2q{j})sin(q{k})')
            features.append(sin_2q[j] * sin_q[k])
    
    # Also include cos(2q_j) crossed with single trig (for completeness)
    for j in joints:
        for k in joints:
            if j == k:
                continue
            feature_names.append(f'cos(2q{j})cos(q{k})')
            features.append(cos_2q[j] * cos_q[k])
            feature_names.append(f'cos(2q{j})sin(q{k})')
            features.append(cos_2q[j] * sin_q[k])
    
    X = np.column_stack(features)
    return X, feature_names


def fit_and_summarize(X, names, y, alpha, label):
    lasso = Lasso(alpha=alpha, max_iter=50000, fit_intercept=False)
    lasso.fit(X, y)
    
    coefs = lasso.coef_
    r2 = lasso.score(X, y)
    n_active = int(sum(abs(c) > 0.01 for c in coefs))
    
    significant = sorted(
        [(n, c) for n, c in zip(names, coefs) if abs(c) > 0.01],
        key=lambda x: -abs(x[1])
    )
    
    # Count by atom type
    n_sin2q_single = sum(1 for n, _ in significant
                        if n.startswith('sin(2q') and ')' == n[-1] and n.count('(') == 1)
    n_sin2q_cross = sum(1 for n, _ in significant
                       if n.startswith('sin(2q') and n.count('(') == 2)
    
    print()
    print('=' * 78)
    print(label)
    print('=' * 78)
    print(f'  Library size:        {len(names)} atoms')
    print(f'  Active atoms:        {n_active} ({n_active/len(names)*100:.1f}%)')
    print(f'  R^2:                 {r2:.6f}')
    print(f'  Residual fraction:   {1-r2:.6f}')
    print(f'  Active sin(2q) singles:    {n_sin2q_single}')
    print(f'  Active sin(2q) crossed:    {n_sin2q_cross}')
    print()
    print('  Top 25 atoms:')
    for name, coef in significant[:25]:
        marker = ' <-- sin(2q)' if 'sin(2q' in name else ''
        print(f'    {coef:+.4f}  {name}{marker}')
    
    return {
        'label': label,
        'library_size': len(names),
        'n_active': n_active,
        'r2': float(r2),
        'n_sin2q_single_active': n_sin2q_single,
        'n_sin2q_cross_active': n_sin2q_cross,
        'top_atoms': [{'name': n, 'coef': float(c)} for n, c in significant[:30]],
        'all_active_atoms': [{'name': n, 'coef': float(c)} for n, c in significant],
    }


def main():
    df = pd.read_csv('benchmarks/franka_mass/franka_M11.csv')
    y = df['M11'].values
    
    print('M_11 Inertia-Tensor-Aware Library Test')
    print(f'Samples: {len(df)}')
    print(f'Hypothesis: missing R^2 comes from sin(2q) terms produced by')
    print(f'off-diagonal inertia tensor components in URDF.')
    
    alpha = 0.001
    
    # Build extended library with sin(2q) terms
    X, names = build_library_with_sin2q(df)
    result = fit_and_summarize(
        X, names, y, alpha,
        'EXTENDED: + sin(2q) singles and cross-products'
    )
    
    # Compare to previous baselines (from earlier scripts)
    baseline_no_triples = 0.851880
    with_triples_only = 0.885730
    
    print()
    print('=' * 78)
    print('COMPARISON ACROSS GRAMMAR EXTENSIONS')
    print('=' * 78)
    print(f'{"Grammar":50s} {"R^2":>8s}')
    print('-' * 78)
    print(f'{"Singles + Pairs + cos(2q)":50s} {baseline_no_triples:>8.4f}')
    print(f'{"+ Triple products":50s} {with_triples_only:>8.4f}')
    print(f'{"+ sin(2q) and cross-products":50s} {result["r2"]:>8.4f}')
    
    delta_vs_baseline = result['r2'] - baseline_no_triples
    delta_vs_triples = result['r2'] - with_triples_only
    print()
    print(f'  Improvement vs baseline:      {delta_vs_baseline:+.4f}')
    print(f'  Improvement vs triples-only:  {delta_vs_triples:+.4f}')
    
    print()
    print('=' * 78)
    print('VERDICT')
    print('=' * 78)
    if result['r2'] >= 0.99:
        print('  sin(2q) atoms CLOSE the gap. Hypothesis 3 (off-diagonal inertia')
        print('  tensors) is confirmed. Grammar extension finalized.')
        print()
        print('  RECOMMENDED PIR-JEPA grammar atoms:')
        print('    - Constants')
        print('    - cos(q_j), sin(q_j)')
        print('    - cos(2q_j), sin(2q_j)')
        print('    - cos/sin(q_j) * cos/sin(q_k)  for j != k')
        print('    - sin(2q_j) * cos/sin(q_k)     for j != k')
    elif result['r2'] >= 0.95:
        print(f'  sin(2q) atoms HELP substantially (R^2 = {result["r2"]:.4f}).')
        print('  Hypothesis 3 is partially confirmed. Remaining residual likely')
        print('  PyBullet numerical noise. Grammar is sufficient for PIR-JEPA.')
    elif delta_vs_baseline > 0.05:
        print(f'  sin(2q) atoms help but do not close the gap (R^2 = {result["r2"]:.4f}).')
        print('  Some structure remains unexplained. Try combining sin(2q) with')
        print('  triple products, or examine PyBullet mass matrix implementation.')
    else:
        print(f'  sin(2q) atoms do NOT meaningfully help (delta = {delta_vs_baseline:+.4f}).')
        print('  Hypothesis 3 is rejected. The structure is not in standard')
        print('  trigonometric polynomial atoms. Need to examine PyBullet output')
        print('  for non-trig structure (e.g., compare with Pinocchio if installable).')
    
    # Save
    output_path = Path('benchmarks/franka_mass/M11_sin2q_analysis.json')
    with open(output_path, 'w') as f:
        json.dump({
            'task': 'M11',
            'samples': len(df),
            'hypothesis': 'Off-diagonal inertia tensor components produce sin(2q) terms',
            'baseline_no_triples_r2': baseline_no_triples,
            'with_triples_only_r2': with_triples_only,
            'extended_with_sin2q': result,
        }, f, indent=2)
    print()
    print(f'Full results saved: {output_path}')


if __name__ == '__main__':
    main()
