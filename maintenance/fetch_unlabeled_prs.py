# noqa: D100
import os
from datetime import datetime

import requests

# Configuration
# Your GitHub Personal Access Token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if GITHUB_TOKEN is None:
    print(
        "Please specify your GitHub Personal Access Token as GITHUB_TOKEN in your .bashrc"
    )
    exit()
GITHUB_USERNAME = "mesa"
GITHUB_REPO = "mesa"
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}
TIMEOUT = 20


def get_latest_release_date() -> str:
    """Fetches the latest release date from the GitHub repository."""
    url = (
        f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/releases/latest"
    )
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()  # Raises an exception for HTTP error codes
    return response.json()["published_at"]


def get_closed_pull_requests_since_latest_release(
    latest_release_date,
) -> list[dict[str, any]]:
    """Fetches pull requests created or updated after the latest release date, then filters by merged date."""
    pull_requests = []
    page = 1
    while True:
        # Fetch PRs that were created or updated after the latest release date
        url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/pulls?state=closed&base=main&sort=updated&direction=desc&page={page}"
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        prs = response.json()
        if not prs:
            break

        # Convert latest release date to datetime for comparison
        latest_release_datetime = datetime.strptime(
            latest_release_date, "%Y-%m-%dT%H:%M:%SZ"
        ).astimezone()

        for pr in prs:
            # Convert PR's `updated_at` to datetime for comparison
            pr_updated_at = datetime.strptime(
                pr["updated_at"], "%Y-%m-%dT%H:%M:%SZ"
            ).astimezone()
            # Stop fetching if PR was updated before the latest release
            if pr_updated_at < latest_release_datetime:
                return pull_requests

            if pr["merged_at"]:
                pr_merged_at = datetime.strptime(
                    pr["merged_at"], "%Y-%m-%dT%H:%M:%SZ"
                ).astimezone()
                if pr_merged_at > latest_release_datetime and not pr["labels"]:
                    pull_requests.append(pr)
        page += 1
    return pull_requests


def main() -> None:  # noqa: D103
    # Based on https://github.com/mesa/mesa/pull/1917#issuecomment-1871352058
    latest_release_date = get_latest_release_date()
    pull_requests = get_closed_pull_requests_since_latest_release(latest_release_date)
    if len(pull_requests) <= 0:
        return
    print("These pull requests must be labeled:")
    for pr in pull_requests:
        print(f"  PR #{pr['number']}: {pr['title']} - Merged at: {pr['merged_at']}")


if __name__ == "__main__":
    main()
