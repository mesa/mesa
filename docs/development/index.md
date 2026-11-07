# Developer guide

This guide is for everyone who works on Mesa itself: contributors, reviewers, and maintainers. It explains how we work together, what we value, and what we have decided in the past. Its meant to help you create a broad understanding what we do and how and why we do them.

New here? Start with the [contributing guide](https://github.com/mesa/mesa/blob/main/CONTRIBUTING.md) for setup and first steps, then come back to this page.

```{toctree}
:hidden: true
:maxdepth: 2

Design principles <principles>
Standards <standards>
Reviewing contributions <review>
Project decisions <decisions>
Roles and releases <maintainers>
```

| Page | Read it when… |
|---|---|
| [Design principles and trade-offs](principles) | You're proposing or designing a change and want to know what Mesa values and which questions we will ask. |
| [Code, test, and documentation standards](standards) | You're implementing a change: style, API conventions, exceptions, determinism, tests, docs, examples, deprecations. |
| [Reviewing contributions](review) | You're reviewing a PR, as a maintainer or as a peer, or want to know what reviewers look for. |
| [Project decisions](decisions) | You want to know whether something has been decided before, and why. |
| [Roles and releases](maintainers) | You want to know how roles in Mesa work, or you're cutting a release. |

## How we work

Mesa aims to provide elegant, scalable, flexible, and powerful building blocks to a wide audience of agent-based modellers. Most of the work in getting there is not writing code. It is agreeing on *what problem we are solving, for whom*, and *what a good solution looks like*, including the trade-offs a library has to make. Code review comes last.

We work in three stages:

1. **Problem**: what are we solving, and for whom?
2. **Concept**: what should a good solution do, and what does it cost?
3. **Implementation**: does the code deliver the agreed solution?

A small, clear bug fix can move through all three in one PR. A new feature or a change to a public API usually needs the first two settled before code is useful. Getting the earlier stages right is what makes the later ones fast and pleasant, for you and for reviewers.

### When to open what

- **Bug fixes**: for clear, scoped bugs, opening a PR directly is usually fine. For larger or uncertain bug fixes, open an issue or discussion first. Show the bug: a minimal reproducer and the error output. Also explain how you ran into it and how it affects real models. Reports of edge cases that were found only by reading the code, and that no realistic model would hit, may be closed.
- **Enhancements and new features**: open an issue or discussion first, agree on direction with maintainers, and wait for explicit approval before opening a PR. Feature PRs without prior alignment may be closed.
- **Questions and open-ended ideas**: use [GitHub Discussions](https://github.com/mesa/mesa/discussions). Issues are for concrete, trackable work.
- **When in doubt**: start with a discussion. It's always easier to move from discussion to implementation than to rework a large PR.

### Stage 1: Problem alignment

Before jumping into solutions, we first align on the problem:

- **Target audience**: who is this feature for? What's their experience level and context?
- **Use cases**: when and why would someone use this? What are they trying to accomplish?
- **Existing solutions**: what do other libraries offer? What can we learn from or connect to?
- **Gap analysis**: given existing solutions (both in Mesa and elsewhere), where are the gaps?

This stage answers the fundamental question: *is this problem worth solving, and are we the right ones to solve it?* Sometimes the answer is that we need better documentation, an example, or a pointer to an existing Python library rather than a new feature.

### Stage 2: Conceptual and high-level design

Once we agree on the problem, we design the solution at a high level:

- **User API**: what will the interface look like from a user's perspective? How will modellers interact with it?
- **Core architecture**: what data structures and patterns will we use internally?
- **Integration**: how does this fit with existing Mesa components?
- **Trade-offs**: which of our [design principles](principles) does this touch, and which side are we choosing?

The goal is to wrap our heads around the conceptual approach before diving into code.

### Stage 3: Implementation

With alignment on both problem and design, implementation can proceed. For enhancements and new features, this stage starts only after maintainers confirm they are open to a PR:

- Write the code following our [standards](standards).
- Include or update tests and check test coverage.
- Include or update documentation and check the rendered result on Read the Docs.
- Open a PR for review.

If needed, also update relevant examples, update or extend the tutorial, or write a section for the [migration guide](../migration_guide).

### Pathfinding PRs

For complex features where feasibility or proportionality is uncertain, we sometimes use *pathfinding PRs*. These are exploratory implementations meant to:

- Test whether an approach is technically viable
- Understand the true complexity and scope
- Gather concrete data (for example micro-benchmarks) to inform design decisions
- Spark discussion with working code rather than abstract ideas

Pathfinding PRs aren't expected to be merge-ready. They're tools for learning and discussion. Say clearly in the PR description that it is a pathfinding PR and what question it is meant to answer. It's good fashion to open them as a [Draft PR](https://github.blog/news-insights/product-news/introducing-draft-pull-requests/).

### Example workflow

1. **Discussion opened**: "I think feature X should be in Mesa."
2. **Problem alignment**: the community discusses target users, use cases, and existing solutions, and identifies gaps.
3. **Conceptual design**: proposals for the user API and architecture are shared and refined.
4. **Pathfinding** (if needed): a quick prototype tests feasibility and gathers data.
5. **Implementation PR**: full implementation with tests, docs, and examples.
6. **Review and iteration**: feedback leads to refinements.
7. **Merge**: the feature becomes part of Mesa.

Figuring out the right direction is often as valuable as implementing a predefined solution. Don't hesitate to ask questions and propose ideas early.

## Using AI tools

AI tools are welcome, for learning, exploring, and writing code. Our [Code of Conduct](https://github.com/mesa/mesa/blob/main/CODE_OF_CONDUCT.md#ai-and-llm-use) sets the expectations: understand what you submit, stay in the driver's seat, and engage meaningfully. In practice:

- **Your reasoning should be your own.** The motivation for a change, the trade-offs you weighed, and what you learned are exactly what reviewers want to hear from *you*. Don't outsource that part.
- **Write your PR and issue text yourself.** Reviewers want to read what you think, in your own words, not generic AI output. Using AI to improve the language of text you wrote is fine; most of us aren't native English speakers.
- **Verify before you claim.** Run the reproducer, run the tests, measure the benchmark. Don't report a bug, a fix, or a speedup you haven't seen yourself.
- **Say how you used AI** in your PR description, briefly.
- **Point your tools at this guide.** The repository's [`AGENTS.md`](https://github.com/mesa/mesa/blob/main/AGENTS.md) tells AI coding assistants how to work on Mesa, including the stages above.
