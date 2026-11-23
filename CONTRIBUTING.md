# Contributing

_For candidates interested in participating in the Google Summer of Code (GSoC), check out Mesa’s [GSoC guide](https://github.com/mesa/mesa/blob/main/docs/GSoC.md)._

Welcome! As an open source project, Mesa welcomes contributions of many forms, from beginners and experts alike: code, bug reports, reviews, documentation, tutorials, examples, answering questions, and sharing Mesa with others. We post our development session agendas and notes on [Mesa discussions], and we have a [Discord] for casual conversation.

No contribution is too small. Contributions can be too big, though, so let's talk first.

## How we work, in short

Most of the work in improving Mesa is agreeing on **what problem we are solving, for whom**, and **what a good solution looks like**. Code comes after that. So:

- **Bug fixes**: for clear, scoped bugs, open a PR directly. Include a minimal reproducer. For larger or unclear bugs, open an issue first.
- **Enhancements and new features**: open an [issue](https://github.com/mesa/mesa/issues) or [discussion](https://github.com/mesa/mesa/discussions) first, and wait for a maintainer to agree on direction before opening a PR.
- **Questions and ideas**: start a [discussion](https://github.com/mesa/mesa/discussions).

It's always easier to move from a discussion to an implementation than to rework a large PR. The [developer guide](https://mesa.readthedocs.io/latest/development/) explains this process in more depth, along with our design principles, standards, and past decisions.

## I have no idea where to start
That's fine! Here's a rough outline where you could start, depending on your experience:

### I'm a modeller (but not an experienced developer)
You already know how to build Mesa models (if not skip below), and probably have found things Mesa can't do (elegantly). You want to improve that. Awesome!

First step is to install some proper tools, if you haven't already.
- A good IDE helps for code development, testing, and formatting. [PyCharm](https://www.jetbrains.com/pycharm/) or [VSCode](https://code.visualstudio.com/) for example.
- Dive into Git and GitHub. Watch some videos, this takes some time to click. [GitHub Desktop](https://desktop.github.com/) is great.
- [`https://github.dev/mesa/mesa`](https://github.dev/mesa/mesa) is great for small changes (to docs).

Learn the tools, talk to us about what you want to change, and open a small PR once direction is clear. For enhancements/new features, get maintainer approval in an issue/discussion first. Or update an [example model](https://github.com/mesa/mesa-examples) (check open [issues](https://github.com/mesa/mesa-examples/issues))!

### I'm a developer (but not a modeller)
Awesome! You have the basics of open source software development (if not check above), but not much modelling experience.

First step is to start thinking like a modeller. To understand the fine details about our library and contribute meaningfully, get some modelling experience:
- Go though our series of introductory tutorials at [Getting Started](https://mesa.readthedocs.io/latest/getting_started.html). While going through them, dive into the source code to really see what everything does.
- Follow an ABM course (if possible). They might be a bit outdated programming language wise, but conceptual they're sound.
  - This MOOC on ABM concepts: [Agent Based Modeling](https://ocw.tudelft.nl/course-lectures/agent-based-modeling/)
  - This MOOC on practical ABM modelling: [Agent-Based Models with Python: An Introduction to Mesa](https://www.complexityexplorer.org/courses/172-agent-based-models-with-python-an-introduction-to-mesa)
- Go though multiple of our [examples](https://github.com/mesa/mesa-examples). Play with them, modify things, and get a feel for Mesa and ABMs.
  - Check our open [issues](https://github.com/mesa/mesa-examples/issues) for the examples.
  - If you see anything you want to improve, feel free to open a (small) PR for bug fixes/docs. For enhancements/new features, discuss first and wait for maintainer approval.
- If you have a feel for Mesa, check our [discussions](https://github.com/mesa/mesa/discussions) and [issues](https://github.com/mesa/mesa/issues).
  - Also go through our [release notes](https://github.com/mesa/mesa/releases) to see what we recently have been working on, and see some examples of successful PRs.
- Once you found or thought of a nice idea, comment on the issue/discussion (or open a new one) and get to work!

### I'm both
That's great! You can start with bug fixes, docs, or tests right away. For enhancements/new features, align in an issue/discussion and wait for maintainer approval before opening a PR. Reach out to us anytime, start small but don't be afraid to dream big!

### I'm neither
Start with creating your own models, for fun. Once you have some experience, move to the topics above.

## Setting up and submitting a change

1. Fork the Mesa repository and [clone your repository] to your machine.
2. Create a branch: `git checkout -b BRANCH_NAME`, and run `git config pull.rebase true` to avoid messy merge commits.
3. Install an editable version with developer requirements: `pip install -e ".[dev]"`.
4. Install the pre-commit hooks, which format and lint your code with [ruff] on every commit: `pip install pre-commit && pre-commit install`.
5. Make your change, including tests and documentation. See the [standards](https://mesa.readthedocs.io/latest/development/standards.html).
6. Run the tests and check coverage didn't go down: `pytest --cov=mesa tests/`.
7. Commit with a meaningful message (see [Tim Pope's guideline]), push to your fork, and [create a pull request].
8. Pick the bug or feature PR template and fill in every section. Link the issue or discussion it addresses.

Make sure your PR passes the [GH Actions build]. If you get stuck on tests, CI, or anything else, ask in the PR. We're happy to help.

## Using AI tools

AI tools are welcome. Our [Code of Conduct](https://github.com/mesa/mesa/blob/main/CODE_OF_CONDUCT.md#ai-and-llm-use) asks that you understand what you submit, stay in the driver's seat, and engage meaningfully. Write your motivation and reasoning in your own words, verify every claim yourself, and mention how you used AI in your PR. If you use an AI coding assistant, it will pick up our [AGENTS.md](AGENTS.md).

## Further reading

- [Developer guide](https://mesa.readthedocs.io/latest/development/): how we work, from problem to concept to code
- [Design principles and trade-offs](https://mesa.readthedocs.io/latest/development/principles.html)
- [Code, test, and documentation standards](https://mesa.readthedocs.io/latest/development/standards.html), including the deprecation policy and examples policy
- [Reviewing contributions](https://mesa.readthedocs.io/latest/development/review.html)
- [Project decisions](https://mesa.readthedocs.io/latest/development/decisions.html)
- [Roles and releases](https://mesa.readthedocs.io/latest/development/maintainers.html)

## Licensing

The license of this project is located in [LICENSE]. By submitting a contribution to this project, you are agreeing that your contribution will be released under the terms of this license.

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
[discord]: https://discord.gg/UUGJvdtEJu
[mesa discussions]: https://github.com/mesa/mesa/discussions
[pep8]: https://www.python.org/dev/peps/pep-0008
[pre-commit]: https://github.com/pre-commit/pre-commit
[tim pope's guideline]: https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
