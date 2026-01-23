# Contributing

_For candidates interested in participating in the Google Summer of Code (GSoC), checkout Mesa’s [GSoC guide](https://github.com/mesa/mesa/blob/main/docs/GSoC.md)._

As an open source project, Mesa welcomes contributions of many forms, and from beginners to experts. If you are
curious or just want to see what is happening, we post our development session agendas
and development session notes on [Mesa discussions]. We also have a threaded discussion forum on [Matrix]
for casual conversation.

In no particular order, examples include:

- Code patches
- Bug reports and patch reviews
- New features
- Documentation improvements
- Tutorials

No contribution is too small. Although, contributions can be too big, so let's
discuss via [Matrix] OR via [an issue].

**To submit a contribution**

- Create a ticket for the item that you are working on.
- Fork the Mesa repository.
- [Clone your repository] from Github to your machine.
- Create a new branch in your fork: `git checkout -b BRANCH_NAME`
- Run `git config pull.rebase true`. This prevents messy merge commits when updating your branch on top of Mesa main branch.
- Install an editable version with developer requirements locally: `pip install -e ".[dev]"`
- Edit the code. Save.
- Git add the new files and files with changes: `git add FILE_NAME`
- Git commit your changes with a meaningful message: `git commit -m "Fix issue X"`
- If implementing a new feature, include some documentation in docs folder.
- Make sure that your submission works with a few of the examples in the examples repository. If adding a new feature to mesa, please illustrate usage by implementing it in an example.
- Make sure that your submission passes the [GH Actions build]. See "Testing and Standards below" to be able to run these locally.
- Make sure that your code is formatted according to [the ruff] standard (you can do it via [pre-commit]).
- Push your changes to your fork on Github: `git push origin NAME_OF_BRANCH`.
- [Create a pull request].
- Describe the change w/ ticket number(s) that the code fixes.
- Format your commit message as per [Tim Pope's guideline].

## I have no idea where to start
That's fine! Here's a rough outline where you could start, depending on your experience:

### I'm a modeller (but not an experienced developer)
You already know how to build Mesa models (if not skip below), and probably have found things Mesa can't do (elegantly). You want to improve that. Awesome!

First step is to install some proper tools, if you haven't already.
- A good IDE helps for code development, testing and formatting. [PyCharm](https://www.jetbrains.com/pycharm/) or [VSCode](https://code.visualstudio.com/) for example.
- Dive into Git and GitHub. Watch some videos, this takes some time to click. [GitHub Desktop](https://desktop.github.com/) is great.
- [`https://github.dev/mesa/mesa`](https://github.dev/mesa/mesa) is great for small changes (to docs).

Learn the tools, talk to us about what you want to change, and open a small PR. Or update an [example model](https://github.com/mesa/mesa-examples) (check open [issues](https://github.com/mesa/mesa-examples/issues))!

### I'm a developer (but not a modeller)
Awesome! You have the basics of open-source software development (if not check above), but not much modelling experience.

First step is to start thinking like a modeller. To understand the fine details about our library and contribute meaningfully, get some modelling experience:
- Go though our series of introductory tutorials at [Getting Started](https://mesa.readthedocs.io/latest/getting_started.html). While going through them, dive into the source code to really see what everything does.
- Follow an ABM course (if possible). They might be a bit outdated programming language wise, but conceptual they're sound.
  - This MOOC on ABM concepts: [Agent Based Modeling](https://ocw.tudelft.nl/course-lectures/agent-based-modeling/)
  - This MOOC on practical ABM modelling: [Agent-Based Models with Python: An Introduction to Mesa](https://www.complexityexplorer.org/courses/172-agent-based-models-with-python-an-introduction-to-mesa)
- Go though multiple of our [examples](https://github.com/mesa/mesa-examples). Play with them, modify things and get a feel for Mesa and ABMs.
  - Check our open [issues](https://github.com/mesa/mesa-examples/issues) for the examples.
  - If you see anything you want to improve, feel free to open a (small) PR!
- If you have a feel for Mesa, check our [discussions](https://github.com/mesa/mesa/discussions) and [issues](https://github.com/mesa/mesa/issues).
  - Also go through our [release notes](https://github.com/mesa/mesa/releases) to see what we recently have been working on, and see some examples of successful PRs.
- Once you found or thought of a nice idea, comment on the issue/discussion (or open a new one) and get to work!

### I'm both
That's great! You can just start working on things, reach out to us. Skim to the list above if you feel you're missing anything. Start small but don't be afraid to dream big!

### I'm neither
Start with creating your own models, for fun. Once you have some experience, move to the topics above.

## Mesa development process
Mesa is a library that aims to provide elegant, scalable, flexible, and powerful building blocks to a wide audience of agent-based modellers. When contributing, it helps to understand our development philosophy and process.

### When to open what
- **Small enhancements or bug fixes**: Opening an issue or directly a PR is often great.
- **Larger features or uncertain scope**: Open a [Discussion](https://github.com/mesa/mesa/discussions) first. This helps align on the problem and approach before investing significant effort.
- **When in doubt**: Start with a Discussion. It's always easier to move from discussion to implementation than to rework a large PR.

### Our development process
For significant features, we typically follow a three-stage process. While not always linear, this flow helps ensure we build the right thing the right way.

#### Stage 1: Problem alignment
Before jumping into solutions, we first align on the problem:

- **Target audience**: Who is this feature for? What's their experience level and context?
- **Use cases**: When and why would someone use this? What are they trying to accomplish?
- **Existing solutions**: What do other libraries offer? What can we learn from or connect to?
- **Gap analysis**: Given existing solutions (both in Mesa and elsewhere), where are the gaps?

This stage answers the fundamental question: *Is this problem worth solving, and are we the right ones to solve it?*

#### Stage 2: Conceptual/high-level design
Once we agree on the problem, we design the solution at a high level:

- **User API**: What will the interface look like from a user's perspective? How will modellers interact with this feature?
- **Core architecture**: What data structures and patterns will we use internally?
- **Integration**: How does this fit with existing Mesa components?

The goal is to wrap our heads around the conceptual approach before diving into code.

Sometimes the answer is that we just need better documentation. Python itself is a powerful tool and there are many existing libraries in the ecosystem to leverage.

#### Stage 3: Implementation
With alignment on both problem and design, implementation can proceed:

- Write the code following Mesa's standards
- Include/update tests and and check test coverage
- Include/update documentation and check Readthedocs rendering
- Open a PR for review

And if needed:
- Update relevant examples
- Update/extend the tutorial
- Write a section for the [migration guide](docs/migration_guide.md)

### Pathfinding PRs
For complex features where feasibility or proportionality is uncertain, we sometimes use *pathfinding PRs*. These are exploratory implementations meant to:

- Test whether an approach is technically viable
- Understand the true complexity and scope
- Gather concrete data (e.g., micro-benchmarks) to inform design decisions
- Spark discussion with working code rather than abstract ideas

Pathfinding PRs aren't expected to be merge-ready. They're tools for learning and discussion. If you're exploring a complex feature, consider posting initial results and ideas in a Discussion before opening a full PR.

### Example workflow
Here's how this process might look in practice:

1. **Discussion opened**: "I'd think feature X should be in Mesa"
2. **Problem alignment**: Community discusses target users, use cases, existing solutions, and identifies gaps
3. **Conceptual design**: Proposals for user API and architecture are shared and refined
4. **Pathfinding** (if needed): A quick prototype tests feasibility and gathers data
5. **Implementation PR**: Full implementation with tests, docs, and examples
6. **Review and iteration**: Community feedback leads to refinements
7. **Merge**: Feature becomes part of Mesa

Figuring out the right direction is often as valuable as implementing a predefined solution. Don't hesitate to ask questions and propose ideas early in the process!

## Testing and Code Standards

```{image} https://codecov.io/gh/mesa/mesa/branch/main/graph/badge.svg
:target: https://codecov.io/gh/mesa/mesa
```

```{image} https://img.shields.io/badge/code%20style-ruff-000000.svg
:target: https://github.com/astral-sh/ruff
```

As part of our contribution process, we practice continuous integration and use GH Actions to help enforce best practices.

If you're changing previous Mesa features, please make sure of the following:

- Your changes pass the current tests.
- Your changes pass our style standards.
- Your changes don't break the models or your changes include updated models.
- Additional features or rewrites of current features are accompanied by tests.
- New features are demonstrated in a model, so folks can understand more easily.

To ensure that your submission will not break the build, you will need to install Ruff and pytest.

```bash
pip install ruff pytest pytest-cov
```

We test by implementing simple models and through traditional unit tests in the tests/ folder. The following only covers unit tests coverage. Ensure that your test coverage has not gone down. If it has and you need help, we will offer advice on how to structure tests for the contribution.

```bash
py.test --cov=mesa tests/
```

With respect to code standards, we follow [PEP8] and the [Google Style Guide]. We use [ruff format] (a more performant alternative to `black`) as an automated code formatter. You can automatically format your code using [pre-commit], which will prevent `git commit` of unstyled code and will automatically apply ruff style so you can immediately re-run `git commit`. To set up pre-commit run the following commands:

```bash
pip install pre-commit
pre-commit install
```

You should no longer have to worry about code formatting. If still in doubt you may run the following command. If the command generates errors, fix all errors that are returned.

```bash
ruff check . --fix
```

## Licensing

The license of this project is located in [LICENSE].  By submitting a contribution to this project, you are agreeing that your contribution will be released under the terms of this license.

## Roles in Mesa
Mesa has several roles to help structure our collaboration and recognize great work. They also form a progression path for community members to that want to take on increasing responsibility in the project. Since we're all volunteers, everyone contributes what they can when they can - there are no minimum time commitments. Also, the best ideas and contributions can come from anyone, these roles are just a way to coordinate our efforts.

Feel free to reach out to us anytime to discuss your interests and ambitions in the project. We're always happy to chat about how you can grow your involvement in Mesa!

### Contributor
Contributors help improve Mesa through:
- Code contributions
- Documentation improvements
- Bug reports and fixes
- Example models
- Tutorial improvements
- Answering questions
- Participating in discussions
- Testing pre-releases
- Sharing Mesa with others

Everyone can contribute what they can, when they can. No contribution is too small! Contributors who have a PR successfully merged receive the "Contributor" label on GitHub.

### Collaborator
When contributors consistently demonstrate technical skills and community mindset through their contributions, they may be invited to become collaborators. Collaborators help coordinate by:

- Reviewing pull requests
- Triaging issues and discussions
- Coordinating between contributors
- Leading specific areas of development
- Helping new contributors
- Participating in project planning
- Building community

The collaborator role recognizes people who help Mesa grow through both their technical contributions and community involvement. Collaborators receive GitHub triage permissions and the "Collaborator" label.

### Maintainer
Maintainers help guide Mesa's overall development while ensuring the project remains sustainable and welcoming. They focus on:
- Project vision and roadmap
- Major architectural decisions
- Release management
- Community governance
- Mentoring collaborators
- Setting community standards
- Long-term sustainability

Maintainers are selected based on their technical expertise, project understanding, and community leadership. Maintainers receive full repository permissions and the "Member" label.

### Special Roles
In some cases, special roles may be created for specific purposes, such as leading particular initiatives or components within Mesa. These roles are created as needed based on project requirements and may come with specific permissions and labels.

All roles are expected to:
- Follow Mesa's code of conduct
- Communicate respectfully
- Work collaboratively
- Help maintain a welcoming community
- Make decisions transparently

Mesa grows through good ideas and contributions. We're all volunteers working together to make Mesa better. Don't hesitate to reach out to any maintainer to discuss your interests and potential growth within the project!

## Mesa examples structure and policy
Mesa maintains a curated set of core examples within the main repository, complemented by a separate [mesa-examples](https://github.com/mesa/mesa-examples) repository for community contributions. This two-tier structure balances maintainability with community creativity.

### Core examples (in `mesa/examples`)
Core examples in the main Mesa repository are classic, well-known agent-based models that demonstrate Mesa's capabilities and serve as learning resources. They are organized into two categories:

- **Basic Examples** use only stable Mesa features and are ideal starting points for beginners.
- **Advanced Examples** are more complex models that may use experimental features to demonstrate advanced concepts.

Core examples are:
- Tested in CI, including batch runs and visualizations
- Maintained to high code quality standards
- Updated to work with each Mesa release
- Documented and included in ReadTheDocs
- Some are used in performance benchmarking

### Community examples (in mesa-examples)
The [mesa-examples](https://github.com/mesa/mesa-examples) repository serves as a gallery for user-contributed models and creative applications of Mesa. We welcome diverse contributions here with more relaxed maintenance requirements. Examples in this repository should include:
- A clear README explaining the model
- Requirements or environment files for reproducibility
- Working code compatible with at least one Mesa major version

We accept contributions liberally to this repository to showcase the breadth of Mesa applications, even if we cannot actively maintain all examples long-term.

### Contributing examples
When contributing a new example, consider:
- **For core examples**: Propose additions through a GitHub discussion first. Core examples should be:
  - widely-recognized canonical models
  - demonstrate specific Mesa features effectively (that are not already sufficiently demonstrated by other models)
- **For community examples**: Open a PR directly to mesa-examples with your model, README, and environment file.
- **Improving existing examples**: PRs to update or enhance any example are always welcome in either repository.

This structure allows us to maintain a stable, high-quality set of learning resources while encouraging community creativity and diverse applications of Mesa.

Historical context and further motivation can be found in discussion [#2330](https://github.com/mesa/mesa/discussions/2330) and PR [#2349](https://github.com/mesa/mesa/pull/2349).

## Maintainers' notes
Some notes useful for Mesa maintainers.

### Releases
To create a new release, follow these steps:

1. Ensure all pull requests (PRs) have a clear title and are labeled with at least one label. Check [this link](https://github.com/mesa/mesa/pulls?q=is%3Apr+is%3Amerged+no%3Alabel+merged%3A%3E%3D2024-03-01+) to see if all PRs are labeled. These labels will be used when drafting the changelog using the [`.github/release.yml`](https://github.com/mesa/mesa/blob/main/.github/release.yml) configuration.
2. Navigate to the [Releases](https://github.com/mesa/mesa/releases) section in the GitHub UI and click the _Draft a new release_ button.
3. Specify the upcoming tag in the _Choose a tag_ and _Release title_ fields (e.g., `v3.0.0`).
   - For pre-releases, add a `a`, `b` or `rc` and a number behind the version tag (see [Versioning](https://packaging.python.org/en/latest/discussions/versioning/)), and check the box _Set as a pre-release_.
4. Use the _Generate release notes_ button to automatically create release notes. Review them carefully for accuracy, and update labels and edit PR titles if necessary (step 1).
5. Write a _Highlights_ section summarizing the most important features or changes in this release.
6. Copy the release notes and save them by clicking the grey _Save draft_ button.
7. Open a new PR to update the version number in [`mesa/__init__.py`](https://github.com/mesa/mesa/blob/main/mesa/__init__.py) and add the copied release notes to the [`HISTORY.md`](https://github.com/mesa/mesa/blob/main/HISTORY.md).
8. Once this PR is merged, return to the _Releases_ section and publish the draft release.
9. The [`release.yml`](https://github.com/mesa/mesa/blob/main/.github/workflows/release.yml) CI workflow should automatically create and upload the package to PyPI. Verify this on [PyPI.org](https://pypi.org/project/mesa/).
10. Finally, after release, open a new PR to update the version number in [`mesa/__init__.py`](https://github.com/mesa/mesa/blob/main/mesa/__init__.py) for the next release (e.g., `"3.1.0.dev"`).

A recorded video of this process is [available here](https://youtu.be/JE44jkegmns).

### Deprecation policy
Mesa follows [Semantic Versioning](https://semver.org/) (SemVer) and has a structured deprecation process to ensure users have time to migrate to new APIs while maintaining backward compatibility.

#### Deprecation process
##### Step 1: Introduce alternative
When implementing a replacement feature:

- Add the new, stable alternative
- Optionally add a `PendingDeprecationWarning` to the old feature (signals intent in source code, hidden by default)

```python
warnings.warn(
    "old_feature() will be deprecated in a future release. "
    "Use new_feature() instead.",
    PendingDeprecationWarning,
    stacklevel=2,
)
```

##### Step 2: Complete prerequisites
Before active deprecation, all of the following must be complete:

1. **Documentation updated**: All relevant docs and tutorials reflect the new approach
2. **Migration guide entry added**: Clear entry in the [Migration Guide](https://github.com/mesa/mesa/blob/main/docs/migration_guide.md) explaining what changed and how to update code
3. **Examples updated**: All example models use the new API

##### Step 3: Active deprecation
The alternative must be available for *at least one minor release* before adding a `FutureWarning`. This warning is visible to all users by default.

```python
warnings.warn(
    "old_feature() is deprecated and will be removed in Mesa X.0. "
    "Use new_feature() instead. "
    "See: https://mesa.readthedocs.io/latest/migration_guide.html#section",
    FutureWarning,
    stacklevel=2,
)
```

##### Step 4: Remove deprecated feature
Remove the old feature in the next major version.

#### Version requirements
| Action | Allowed in |
|--------|------------|
| Add alternative + `PendingDeprecationWarning` | Any release |
| Add `FutureWarning` | Minor release (patch only with compelling reason) |
| Remove deprecated feature | Major release only |

#### Experimental features
For features in `mesa.experimental`, steps 1–3 can be done simultaneously, and step 4 can occur in any subsequent release (including minor/patch). Experimental features:

- Can be changed or removed in any release
- Don't require a deprecation warning period
- Should still communicate changes through release notes

Following the full process is encouraged when feasible.

#### Migration guide entry format
Place new entries at the top (newest first). Include:

1. Clear heading describing the change
2. Brief explanation of what changed and why
3. Before/after code examples
4. Links to relevant PRs/issues

````markdown
## Mesa X.Y.0
### Feature Name Change
Brief description of what changed and why.

```python
# Old
old_method()

# New
new_method()
```

- Ref: [PR #1234](https://github.com/mesa/mesa/pull/1234), [Documentation](link)
````

## Special Thanks

A special thanks to the following projects who offered inspiration for this contributing file.

- [Django]
- [18F's FOIA]
- [18F's Midas]

[18f's foia]: https://github.com/18F/foia-hub/blob/master/CONTRIBUTING.md
[18f's midas]: https://github.com/18F/midas/blob/devel/CONTRIBUTING.md
[an issue]: https://github.com/mesa/mesa/issues
[ruff]: https://github.com/astral-sh/ruff
[clone your repository]: https://help.github.com/articles/cloning-a-repository/
[create a pull request]: https://help.github.com/articles/creating-a-pull-request/
[django]: https://github.com/django/django/blob/master/CONTRIBUTING.rst
[gh actions build]: https://github.com/mesa/mesa/actions/workflows/build_lint.yml
[google style guide]: https://google.github.io/styleguide/pyguide.html
[license]: https://github.com/mesa/mesa/blob/main/LICENSE
[matrix]: https://matrix.to/#/#project-mesa:matrix.org`
[mesa discussions]: https://github.com/mesa/mesa/discussions
[pep8]: https://www.python.org/dev/peps/pep-0008
[pre-commit]: https://github.com/pre-commit/pre-commit
[tim pope's guideline]: https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
