"""
M_11 Triple-Product Structural Analysis

Tests whether adding triple-product trigonometric terms to the principled
rigid-body feature library closes the R^2 gap from 0.85 to >= 0.99 on M_11.

Library contents:
  - Constant
  - Singles:    cos(q_j), sin(q_j) for j in {2..7}
  - Doubles:    cos/sin(q_j) * cos/sin(q_k) for j < k in {2..7}
  - Double-angle: cos(2*q_j) for j in {2..7}
  - Triples:    cos/sin(q_j) * cos/sin(q_k) * cos/sin(q_l) for j < k < l in {2..7}

Why distinct joints in triples: same-joint repeats reduce to lower-order atoms
via cos^2 = (1+cos(2q))/2 and sin^2 = (1-cos(2q))/2, which are already in the
library as cos(2q_j) double-angle terms.

Output:
  - R^2 with and without triple products (the key comparison)
  - Top atoms in the extended library
  - Diagnostic: how many triple atoms are needed
  - JSON dump for downstream use
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from itertools import combinations, product
import json
from pathlib import Path


def build_library(df, include_triples=False, joint_indices=range(2, 8)):
    """Build feature library for M_11.
    
    Args:
        df: DataFrame with q1..q7 columns
        include_triples: if True, add triple products of distinct joints
        joint_indices: which joints to include (default 2..7 since M_11 should
                       not depend on q1 by serial-chain structure)
    
    Returns:
        X: (n_samples, n_features) array
        feature_names: list of n_features atom names
    """
    joints = list(joint_indices)
    n_samples = len(df)
    
    feature_names = ['const']
    features = [np.ones(n_samples)]
    
    # Cache trig values
    cos_q = {j: np.cos(df[f'q{j}'].values) for j in joints}
    sin_q = {j: np.sin(df[f'q{j}'].values) for j in joints}
    
    # Singles
    for j in joints:
        feature_names += [f'cos(q{j})', f'sin(q{j})']
        features += [cos_q[j], sin_q[j]]
    
    # Double-angle terms (equivalent to cos^2/sin^2)
    for j in joints:
        feature_names += [f'cos(2q{j})']
        features += [np.cos(2*df[f'q{j}'].values)]
    
    # Pairwise products (distinct joints)
    for j, k in combinations(joints, 2):
        for fj_name, fj_val in [('cos', cos_q[j]), ('sin', sin_q[j])]:
            for fk_name, fk_val in [('cos', cos_q[k]), ('sin', sin_q[k])]:
                feature_names.append(f'{fj_name}(q{j}){fk_name}(q{k})')
                features.append(fj_val * fk_val)
    
    # Triple products (distinct joints) - the new addition
    if include_triples:
        for j, k, l in combinations(joints, 3):
            for (fj_name, fj_val), (fk_name, fk_val), (fl_name, fl_val) in product(
                [('cos', cos_q[j]), ('sin', sin_q[j])],
                [('cos', cos_q[k]), ('sin', sin_q[k])],
                [('cos', cos_q[l]), ('sin', sin_q[l])],
            ):
                feature_names.append(
                    f'{fj_name}(q{j}){fk_name}(q{k}){fl_name}(q{l})'
                )
                features.append(fj_val * fk_val * fl_val)
    
    X = np.column_stack(features)
    return X, feature_names


def fit_and_report(X, feature_names, y, alpha, label):
    """Fit Lasso and report key metrics."""
    lasso = Lasso(alpha=alpha, max_iter=50000, fit_intercept=False)
    lasso.fit(X, y)
    
    coefs = lasso.coef_
    r2 = lasso.score(X, y)
    n_active = int(sum(abs(c) > 0.01 for c in coefs))
    
    significant = sorted(
        [(n, c) for n, c in zip(feature_names, coefs) if abs(c) > 0.01],
        key=lambda x: -abs(x[1])
    )
    
    print(f'\n{"=" * 78}')
    print(f'{label}')
    print(f'{"=" * 78}')
    print(f'  Library size:       {len(feature_names)} atoms')
    print(f'  Active atoms:       {n_active} ({n_active / len(feature_names) * 100:.1f}%)')
    print(f'  R^2:                {r2:.6f}')
    print(f'  Residual SS / Total SS: {1 - r2:.6f}')
    print()
    print(f'  Top 20 atoms by magnitude:')
    for name, coef in significant[:20]:
        sign = ' ' if coef >= 0 else ''
        print(f'     {sign}{coef:+.4f}  {name}')
    
    # Classify active atoms by order
    n_const = sum(1 for n, _ in significant if n == 'const')
    n_single = sum(1 for n, _ in significant
                   if n.startswith(('cos(q', 'sin(q')) and n.count('(') == 1
                   and 'cos(2q' not in n)
    n_double_angle = sum(1 for n, _ in significant if 'cos(2q' in n)
    n_pair = sum(1 for n, _ in significant if n.count('(') == 2)
    n_triple = sum(1 for n, _ in significant if n.count('(') == 3)
    
    print()
    print(f'  Active atoms by order:')
    print(f'    Constants:        {n_const}')
    print(f'    Singles:          {n_single}')
    print(f'    Double-angle:     {n_double_angle}')
    print(f'    Pairwise:         {n_pair}')
    print(f'    Triple:           {n_triple}')
    
    return {
        'label': label,
        'library_size': len(feature_names),
        'n_active': n_active,
        'r2': float(r2),
        'breakdown': {
            'const': n_const,
            'single': n_single,
            'double_angle': n_double_angle,
            'pairwise': n_pair,
            'triple': n_triple,
        },
        'top_atoms': [
            {'name': n, 'coef': float(c)} for n, c in significant[:30]
        ],
        'all_active_atoms': [
            {'name': n, 'coef': float(c)} for n, c in significant
        ],
    }


def main():
    # Load M_11 data
    df = pd.read_csv('benchmarks/franka_mass/franka_M11.csv')
    y = df['M11'].values
    
    print('M_11 Triple-Product Structural Analysis')
    print(f'Samples: {len(df)}')
    print(f'M_11 range: [{y.min():.4f}, {y.max():.4f}]')
    print(f'M_11 mean +/- std: {y.mean():.4f} +/- {y.std():.4f}')
    print(f'Relative variation: {y.std() / y.mean() * 100:.1f}%')
    
    # Lasso regularization strength
    alpha = 0.001
    
    # Baseline: principled library WITHOUT triples (matches earlier analysis)
    X_base, names_base = build_library(df, include_triples=False)
    result_base = fit_and_report(
        X_base, names_base, y, alpha,
        'BASELINE: Singles + Pairs + Double-angle (no triples)'
    )
    
    # Extended: principled library WITH triples
    X_ext, names_ext = build_library(df, include_triples=True)
    result_ext = fit_and_report(
        X_ext, names_ext, y, alpha,
        'EXTENDED: + Triple products (distinct joints)'
    )
    
    # Headline comparison
    print()
    print('=' * 78)
    print('HEADLINE COMPARISON')
    print('=' * 78)
    delta_r2 = result_ext['r2'] - result_base['r2']
    delta_residual = (1 - result_base['r2']) - (1 - result_ext['r2'])
    pct_residual_explained = (
        delta_residual / (1 - result_base['r2']) * 100
        if (1 - result_base['r2']) > 1e-10 else 0.0
    )
    
    print(f'  R^2 without triples:     {result_base["r2"]:.6f}')
    print(f'  R^2 with triples:        {result_ext["r2"]:.6f}')
    print(f'  Delta R^2:                {delta_r2:+.6f}')
    print(f'  Residual variance reduction: {pct_residual_explained:.1f}%')
    print()
    print(f'  Triple atoms now active: {result_ext["breakdown"]["triple"]}')
    print(f'  Library grew from {result_base["library_size"]} to {result_ext["library_size"]} atoms')
    print(f'  Active grew from {result_base["n_active"]} to {result_ext["n_active"]} atoms')
    
    # Verdict
    print()
    print('=' * 78)
    print('VERDICT')
    print('=' * 78)
    if result_ext['r2'] >= 0.99:
        print('  Triple products CLOSE the gap. Grammar = singles + pairs + triples')
        print('  + double-angle is sufficient for M_11.')
        print('  RECOMMENDATION: extend PIR-JEPA grammar with triple-product atoms.')
    elif result_ext['r2'] >= 0.95:
        print(f'  Triple products SUBSTANTIALLY help (R^2 = {result_ext["r2"]:.4f}).')
        print('  Some residual structure remains - possibly URDF inertia tensors')
        print('  or numerical noise from PyBullet. Likely sufficient for PIR-JEPA.')
    elif delta_r2 > 0.05:
        print(f'  Triple products help (delta R^2 = {delta_r2:+.4f}) but do not')
        print('  close the gap. Consider quadruple products or non-trig atoms.')
    else:
        print(f'  Triple products do NOT help (delta R^2 = {delta_r2:+.4f}).')
        print('  The residual is not in trigonometric polynomials of joint angles.')
        print('  Investigate URDF inertia handling in PyBullet.')
    
    # Save results
    output = {
        'task': 'M11',
        'samples': len(df),
        'baseline': result_base,
        'extended': result_ext,
        'delta_r2': float(delta_r2),
        'pct_residual_explained_by_triples': float(pct_residual_explained),
    }
    
    output_path = Path('benchmarks/franka_mass/M11_triple_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print()
    print(f'Full results saved: {output_path}')


if __name__ == '__main__':
    main()
