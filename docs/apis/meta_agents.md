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

## Gating membership changes

**What is it?**

By default, joining or leaving a meta-agent is unconditional — call
`add_member` or `remove_member`, and it happens immediately, every time.
Gating membership changes means adding a condition that must be satisfied
before a join or a leave is allowed to actually happen.

**Why is it important?**

This mirrors how real organizations work. A squad doesn't accept every
recruit automatically. A company doesn't let someone quit mid-project with
no process. A college doesn't admit every applicant. Joining and leaving are
often *decisions*, not just structural changes — and a model that only
supports unconditional membership can't represent that.

**How can we implement it?**

`add_member` and `remove_member` stay simple and unconditional, so approval
logic is added through `meta_methods` instead — a way to bind your own
custom methods onto a specific group when you create it. The pattern is:
write a method that checks your own condition, and only calls
`add_member`/`remove_member` if that condition passes.

**Implementation**

```python
def request_join(self, member, relation="member"):
    if not self.can_accept(member, relation):
        return False
    self.model.meta_agents.add_member(self, member, relation)
    return True

def request_leave(self, member):
    if not self.can_release(member):
        return False
    self.model.meta_agents.remove_member(self, member)
    return True

squad = model.meta_agents.create(
    "Squad", [commander], Agent,
    meta_methods={"request_join": request_join, "request_leave": request_leave},
)
squad.max_size = 10
squad.mission_active = False

squad.request_join(recruit)   # only succeeds if can_accept() allows it
squad.request_leave(soldier)  # only succeeds if can_release() allows it
```

`can_accept` and `can_release` are ordinary methods you write yourself — a
capacity check, a minimum score, a rule that says "no leaving mid-mission,"
or anything else your model needs. Because `request_join`/`request_leave`
are bound separately to each group, different groups in the same model can
each enforce their own, completely different rules.