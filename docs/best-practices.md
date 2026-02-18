# Best Practices

Here are some general principles that have proven helpful for developing models.

## Model Layout

A model should be contained in a folder named with lower-case letters and
underscores, such as `wolf_sheep`. Within that directory:

- `Readme.md` describes the model, how to use it, and any other details.
- `model.py` should contain the model class.
- `agents.py` should contain the agent class(es).
- `app.py` should contain the Solara-based visualization code (optional).

You can add more files as needed, for example:
- `run.py` could contain the code to run the model.
- `analysis.py` could contain any analysis code.

Input data can be stored in a `data` directory, output data in an `output`, processed results in a `results` directory, images in an `images` directory, etc.

All our [examples](examples) follow this layout.

## Randomization

If your model involves some random choice, you can use the built-in `random`
property that many Mesa objects have, including `Model`, `Agent`, and `AgentSet`. This works exactly
like the built-in `random` library.

```python
class AwesomeModel(Model):
  # ...

  def cool_method(self):
    interesting_number = self.random.random()
    print(interesting_number)

class AwesomeAgent(Agent):
  # ...
  def __init__(self, unique_id, model, ...):
    super().__init__(unique_id, model)
    # ...

  def my_method(self):
    random_number = self.random.randint(0, 100)
```

`Agent.random` is just a convenient shorthand in the Agent class to `self.model.random`. If you create your own `AgentSet`
instances, you have to pass `random` explicitly. Typically, you can simply do, in a Model instance,
`my_agentset = AgentSet([], random=self.random)`. This ensures that `my_agentset` uses the same random
number generator as the rest of the model.


When a model object is created, its random property is automatically seeded
with the current time. The seed determines the sequence of random numbers; if
you instantiate a model with the same seed, you will get the same results.
To allow you to set the seed, make sure your model has a `seed` argument in its
`__init__`.

```python
class AwesomeModel(Model):

  def __init__(self, seed=None):
    super().__init__(seed=seed)
    ...

  def cool_method(self):
    interesting_number = self.random.random()
    print(interesting_number)

>>> model0 = AwesomeModel(seed=0)
>>> model0._seed
0
>>> model0.cool_method()
0.8444218515250481
>>> model1 = AwesomeModel(seed=0)
>>> model1.cool_method()
0.8444218515250481
```
