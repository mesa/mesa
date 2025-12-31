# The Emperor's Dilemma Agent-Based Model

This is a Python implementation of the Agent-Based Model (ABM) described in:

[Centola, D., Willer, R., & Macy, M. (2005). The Emperor’s Dilemma: A Computational Model of Self‐Enforcing Norms1. American Journal of Sociology.](https://www.journals.uchicago.edu/doi/10.1086/427321)

The model simulates how unpopular norms can dominate a society even when the vast majority of individuals privately reject them. It demonstrates the "illusion of consensus" where agents, driven by a fear of appearing disloyal, not only comply with a rule they hate but also aggressively enforce it on their neighbors. This phenomenon creates a "trap" of False Enforcement, where the loudest defenders of a norm are often its secret opponents.

The model tracks three key metrics:

1. *Compliance:* The fraction of agents publicly obeying the norm.
2. *Enforcement:* The fraction of agents punishing deviants.
3. *False Enforcement:* The fraction of agents who privately hate the norm but enforce it anyway to signal sincerity.

## Installation

```bash
pip install mesa matplotlib pandas altair
```

## How to Run

```bash
solara run app.py
```