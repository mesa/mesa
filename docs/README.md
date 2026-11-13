# Docs for Mesa

The readable version of the docs is hosted at [mesa.readthedocs.org](http://mesa.readthedocs.org/).

This folder contains the docs that build the docs for the core mesa code on readthedocs.

## How to publish updates to the docs

Updating docs can be confusing. Here are the basic setups.

#### Submit a pull request with updates
1. Create branch (either via branching or fork of repo) -- try to use a descriptive name.
    * `git checkout -b doc-updates`
1. Update the docs. Save.
1. Install the docs dependencies, from the root of the repository.
    * `pip install -e ".[docs]"`
1. Build the docs, from the inside of the docs folder.
    * `make html`
    * The HTML pages are written to `docs/_build/html`. Open `index.html` in a browser.
1. Commit the changes. If there are new files, you will have to explicit add them.
    * `git commit -am "Updating docs."`
1. Push the branch
    * `git push origin doc-updates`
1. From here you will want to submit a pull request to main.

#### Update read the docs

From this point, you will need to find someone that has access to readthedocs. Currently, that is [@jackiekazil](https://github.com/jackiekazil), [@rht](https://github.com/rht), and [@tpike3](https://github.com/tpike3).

1. Accept the pull request into main.
1. Log into readthedocs and launch a new build -- builds take about 10 minutes or so.

### Helpful Sphinx tips
* Build html from docs:
  * `make html`
* Autogenerate / update sphinx from docstrings (replace your name as the author):
  * `sphinx-apidoc -A "Jackie Kazil" -F -o docs mesa/`