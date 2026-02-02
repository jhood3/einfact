# NNEinFact
**Near-Universal Multiplicative Updates for Nonnegative Einsum Factorization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

NNEinFact is a general-purpose library for fitting any nonnegative tensor factorization expressible as an `einsum` contraction. By leveraging a majorization-minimization framework, it provides theoretically guaranteed convergence for a wide family of loss functions, including the $(\alpha, \beta)$-divergence.

### Key Features
* **Modeling**: Fit CP, Tucker, Tensor-Train, or custom models using simple string notation.
* **Loss Functions**: The current implementation supports any $(\alpha, \beta)$ divergence other than $$\alpha = 0, \beta \neq 1$$. Examples include Euclidean distance $$(\alpha=1.0, \beta =1.0$$, KL divergence $$(\alpha=1.0, \beta = 0.0$$, Hellinger distance $$(\alpha=\beta = 0.5)$$, among others. 

For faster training, we recommend leveraging GPUs whenever possible. 

### Quick Start
```python
from einfact import NNEinFact

Y = np.load(...)

shape_dict = {**dict(zip(model_str.split('->')[-1], Y.shape)), 'k': 6, 'r': 10}
# Define a custom model: wk,hk,dk,ikr,jkr -> whdij
model = NNEinFact(
    model_str='wk,hk,dk,ikr,jkr->whdij',
    alpha=0.5, beta=0.5, shape_dict=shape_dict, device='cuda'
)


# Fit to your data
model.fit(Y)

# Retrieve factors
factors = model.get_params()

