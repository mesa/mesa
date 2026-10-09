# Meta-agents

Meta-agents are agents composed of other agents, allowing you to build models
with emergent, multi-level complexity. A built-in membership manager tracks the
network edges and nested relationships between all agents.

To aid in the development of complex simulations, the meta-agent module
includes features for intuitive use, management, and exploration:

- create groups of agents
- add and remove members
- look up who belongs to a group and which groups an agent belongs to
- walk nested levels (`at_level`)
- deactivate or dissolve a group
- find candidate groups (`find_combinations`)

```
        model.meta_agents
       (membership manager)
                |
   create, add_member, remove_member
                |
     memberships (agent → group)
                |
  members_of / groups_of / at_level
                |
         deactivate / dissolve
```

Install the membership manager on the model, then create groups and change
memberships only through `model.meta_agents`.

```python
from mesa import Agent, Model
from mesa.meta_agents import MetaAgents

model = Model()
model.meta_agents = MetaAgents(model)

alice = Agent(model)
bob = Agent(model)
team = model.meta_agents.create("Team", [alice, bob])  # create a group of agents

model.meta_agents.add_member("Team", Agent(model))    # add members
model.meta_agents.members_of("Team")                  # who belongs to a group
model.meta_agents.groups_of(alice)                    # which groups an agent belongs to
model.meta_agents.query_memberships(alice)            # full membership snapshot for one entity
model.meta_agents.at_level(1, root=team)              # walk nested levels
model.meta_agents.dissolve(team)                      # dissolve a group
model.meta_agents.find_combinations([alice, bob], evaluation_func=score)  # candidate groups
```

## Group creation and reuse

Groups are identified by their class name. `create` reuses an existing group
only when at least one of the given agents already belongs to a group with
that class name; the given agents are then added to it. Otherwise a new group
is created. Calling `create` again with the same name but no overlapping
members therefore creates a second, distinct group with the same name (which
makes name-based lookups such as `members_of("Team")` ambiguous). To force a
new group, use a unique class name (e.g., append a timestamp or UUID).

The group agent defaults to a plain `Agent` subclass; pass `mesa_agent_type`
to base it on a custom `Agent` subclass instead.

```python
team = model.meta_agents.create("Team", [alice, bob])

# reuse: alice is already in the group, so carol is added to the same team
team2 = model.meta_agents.create("Team", [carol, alice])
assert team is team2

# more explicit alternative:
model.meta_agents.add_member("Team", carol)

# same name, no overlapping members -> a second, distinct group
other_team = model.meta_agents.create("Team", [dave])
assert other_team is not team

# unique names force distinct groups
team_a = model.meta_agents.create("Team_2026_A", [alice])
team_b = model.meta_agents.create("Team_2026_B", [dave])
```

```{eval-rst}
.. automodule:: mesa.meta_agents
   :members:
   :imported-members:
```
