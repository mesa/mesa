# Roles and releases

How roles in Mesa work, and notes for maintainers.

## Roles
Mesa has several roles to help structure our collaboration and recognize great work. They also form a progression path for community members who want to take on increasing responsibility in the project. Since we're all volunteers, everyone contributes what they can when they can - there are no minimum time commitments. Also, the best ideas and contributions can come from anyone, these roles are just a way to coordinate our efforts.

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
- Follow Mesa's [Code of Conduct](https://github.com/mesa/mesa/blob/main/CODE_OF_CONDUCT.md)
- Communicate respectfully
- Work collaboratively
- Help maintain a welcoming community
- Make decisions transparently

Mesa grows through good ideas and contributions. We're all volunteers working together to make Mesa better. Don't hesitate to reach out to any maintainer to discuss your interests and potential growth within the project!

## Releases

To create a new release, follow these steps:

1. Ensure all pull requests (PRs) have a clear title and are labeled with at least one label. Check [this link](https://github.com/mesa/mesa/pulls?q=is%3Apr+is%3Amerged+no%3Alabel+merged%3A%3E%3D2024-03-01+) to see if all PRs are labeled. These labels will be used when drafting the changelog using the [`.github/release.yml`](https://github.com/mesa/mesa/blob/main/.github/release.yml) configuration.
2. Navigate to the [Releases](https://github.com/mesa/mesa/releases) section in the GitHub UI and click the _Draft a new release_ button.
3. Specify the upcoming tag in the _Choose a tag_ and _Release title_ fields (e.g., `v3.0.0`).
   - For pre-releases, add a `a`, `b`, or `rc` and a number behind the version tag (see [Versioning](https://packaging.python.org/en/latest/discussions/versioning/)), and check the box _Set as a pre-release_.
4. Use the _Generate release notes_ button to automatically create release notes. Review them carefully for accuracy, and update labels and edit PR titles if necessary (step 1).
5. Write a _Highlights_ section summarizing the most important features or changes in this release.
6. Copy the release notes and save them by clicking the grey _Save draft_ button.
7. Open a new PR to update the version number in [`mesa/__init__.py`](https://github.com/mesa/mesa/blob/main/mesa/__init__.py) and add the copied release notes to the [`HISTORY.md`](https://github.com/mesa/mesa/blob/main/HISTORY.md). For stable releases, `docs/_static/switcher.json` is updated automatically when the release tag is pushed.
8. Once this PR is merged, return to the _Releases_ section and publish the draft release.
9. The [`release.yml`](https://github.com/mesa/mesa/blob/main/.github/workflows/release.yml) CI workflow should automatically create and upload the package to PyPI. Verify this on [PyPI.org](https://pypi.org/project/mesa/).
10. Finally, after release, open a new PR to update the version number in [`mesa/__init__.py`](https://github.com/mesa/mesa/blob/main/mesa/__init__.py) for the next release (e.g., `"3.1.0.dev"`).

A recorded video of this process is [available here](https://youtu.be/JE44jkegmns).
