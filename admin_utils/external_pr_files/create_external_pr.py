"""
Python tool for synchronization between source and target repositories.
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, Optional

from logging518.config import fileConfig
from quality_control.cli_unifier import _run_console_tool, handles_console_error
from quality_control.console_logging import get_child_logger
from quality_control.quality_control_parser import QualityControlArgumentsParser

from admin_utils.constants import TRACKED_JSON_PATH

logger = get_child_logger(__file__)


class SyncArgumentParser(QualityControlArgumentsParser):
    """
    Parser that gets args for sync tool
    """

    repo_name: str
    pr_number: str


@dataclass(slots=True)
class CommitConfig:
    """
    Storage for commit data
    """

    repo_path: str
    branch_name: str
    repo_name: str
    pr_number: str
    json_changed: bool
    files_to_sync_found: bool


@dataclass(slots=True)
class SyncConfig:
    """
    Storage for final PR data
    """

    target_repo: str
    changed_files: list[str]
    json_content: Optional[dict]
    json_changed: bool
    commit_sha: str


@dataclass(slots=True)
class SyncResult:
    """
    Result of synchronization operation
    """

    has_changes: bool
    files_to_sync_found: bool
    json_changed: bool


# Wrappers for basic commands
@handles_console_error(ok_codes=(0, 1))
def run_git(args: list[str], **kwargs: str) -> tuple[str, str, int]:
    """
    Run git command via imported function

    Args:
        args (list[str]): Arguments for git command.
        kwargs (str): Keyword arguments.

    Returns:
        tuple[str, str, int]: Result of git command.
    """
    return _run_console_tool("git", args, **kwargs)


@handles_console_error(ok_codes=(0, 1))
def run_gh(args: list[str]) -> tuple[str, str, int]:
    """
    Run gh command via imported function

    Args:
        args (list[str]): Arguments for gh command.

    Returns:
        tuple[str, str, int]: Result of gh command.
    """
    return _run_console_tool("gh", args)


@handles_console_error(ok_codes=(0,))
def run_mkdir(args: list[str], **kwargs: str) -> tuple[str, str, int]:
    """
    Create directory via imported function

    Args:
        args (list[str]): Arguments for mkdir command.
        kwargs (str): Keyword arguments.

    Returns:
        tuple[str, str, int]: Result of mkdir command.
    """
    return _run_console_tool("mkdir", args, **kwargs)


@handles_console_error(ok_codes=(0,))
def run_rm(args: list[str]) -> tuple[str, str, int]:
    """
    Remove anything via imported function

    Args:
        args (list[str]): Arguments for rm command.

    Returns:
        tuple[str, str, int]: Result of rm command.
    """
    return _run_console_tool("rm", args)


@handles_console_error(ok_codes=(0,))
def run_sleep(args: list[str]) -> tuple[str, str, int]:
    """
    Run sleep command via imported function

    Args:
        args (list[str]): Arguments for sleep command.

    Returns:
        tuple[str, str, int]: Result of sleep command.
    """
    return _run_console_tool("sleep", args)


def get_pr_data(repo_name: str, pr_number: str) -> dict[str, Any]:
    """
    Get PR data via gh

    Args:
        repo_name (str): Name of source repo.
        pr_number (str): Number of needed PR in source repo.

    Returns:
        dict[str, Any]: PR data.
    """
    stdout, stderr, return_code = run_gh(
        [
            "pr",
            "view",
            pr_number,
            "--repo",
            repo_name,
            "--json",
            "files,commits",
        ]
    )

    if return_code != 0 or not stdout:
        logger.warning("Failed to get PR data: %s", stderr)
        return {}

    data = json.loads(stdout)
    return cast(dict[str, Any], data)


def check_branch_exists(branch_name: str, repo_path: str = ".") -> bool:
    """
    Check if branch in remote repo exists

    Args:
        branch_name (str): Name of needed branch.
        repo_path (str, optional): Path to repo. Defaults to ".".

    Returns:
        bool: True if needed branch exists in remote repo.
    """
    _, _, return_code = run_git(
        ["show-ref", "--quiet", f"refs/remotes/origin/{branch_name}"], cwd=repo_path
    )
    return bool(return_code == 0)


def clone_repo(target_repo: str, gh_token: str) -> None:
    """
    Clone target repo

    Args:
        target_repo (str): Name of target repo.
        gh_token (str): Token to process operations.
    """
    target_path = Path(target_repo)
    if target_path.exists():
        run_rm(["-rf", str(target_path)])

    run_git(["clone", f"https://{gh_token}@github.com/fipl-hse/{target_repo}.git"])


def setup_git_config(repo_path: str) -> None:
    """
    Setup config

    Args:
        repo_path (str): Path to repo.
    """
    run_git(["config", "user.name", "github-actions[bot]"], cwd=repo_path)
    run_git(
        ["config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"],
        cwd=repo_path,
    )


def check_and_create_label(target_repo: str) -> None:
    """
    Check if label exists or create it

    Args:
        target_repo (str): Path to repo.
    """
    stdout, stderr, return_code = run_gh(
        ["label", "list", "--repo", f"fipl-hse/{target_repo}", "--json", "name"]
    )

    if return_code != 0:
        logger.warning("Failed to get labels: %s", stderr)
        return

    labels = json.loads(stdout) if stdout else []
    label_exists = any(label.get("name") == "automated pr" for label in labels)

    if not label_exists:
        run_gh(
            [
                "label",
                "create",
                "automated pr",
                "--color",
                "0E8A16",
                "--description",
                "Automated pull request",
                "--repo",
                f"fipl-hse/{target_repo}",
            ]
        )
        run_sleep("2")


def checkout_or_create_branch(branch_name: str, repo_path: str) -> None:
    """
    Checkout on existing branch or create it

    Args:
        branch_name (str): Name of needed branch.
        repo_path (str): Path to repo.
    """
    if check_branch_exists(branch_name, repo_path):
        run_git(["checkout", branch_name], cwd=repo_path)
        run_git(["pull", "origin", branch_name], cwd=repo_path)
    else:
        run_git(["checkout", "-b", branch_name], cwd=repo_path)


def add_remote_and_fetch(remote_name: str, repo_url: str, repo_path: str) -> None:
    """
    Add remote and fetch.

    Args:
        remote_name (str): Name of remote repo.
        repo_url (str): Link to remote repo.
        repo_path (str): Path to remote repo.
    """
    stdout, _, _ = run_git(["remote"], cwd=repo_path)
    remotes = stdout.split()

    if remote_name not in remotes:
        run_git(["remote", "add", remote_name, repo_url], cwd=repo_path)

    run_git(["fetch", remote_name], cwd=repo_path)


def get_and_update_json_if_changed(
    repo_path: str, commit_sha: str, changed_files: list[str]
) -> tuple[Optional[dict], bool]:
    """
    Get json content from specific commit

    Args:
        repo_path (str): Path to repo.
        commit_sha (str): Commit SHA.
        changed_files (list[str]): Paths to changed files.

    Returns:
        tuple[Optional[dict], bool]: JSON content from remote branch.
    """
    json_content = None
    json_changed = TRACKED_JSON_PATH in changed_files

    if json_changed:
        stdout, _, return_code = run_git(
            ["show", f"{commit_sha}:{TRACKED_JSON_PATH}"],
            cwd=repo_path,
        )

        if return_code == 0 and stdout:
            json_content = json.loads(stdout)

            TRACKED_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

            with open(TRACKED_JSON_PATH, "w", encoding="utf-8") as f:
                f.write(stdout)

            run_git(["add", TRACKED_JSON_PATH], cwd=repo_path)

    else:
        if TRACKED_JSON_PATH.exists():
            with open(TRACKED_JSON_PATH, "r", encoding="utf-8") as f:
                json_content = json.load(f)
        else:
            logger.warning("Tracked JSON file not found in target repo: %s", TRACKED_JSON_PATH)

    return json_content, json_changed


def get_sync_mapping(json_content: Optional[dict]) -> list[tuple[str, ...]]:
    """
    Extract sync mapping from JSON.

    Args:
        json_content (Optional[dict]): Content of JSON file.

    Returns:
        list[tuple[str, ...]]: Mapping of source/target files from JSON.
    """
    sync_mapping: list[tuple[str, ...]] = []

    if not json_content:
        return []

    for item in json_content:
        source = item.get("source")
        target = item.get("target")
        if source and target:
            sync_mapping.append((source, target))
    return sync_mapping


def sync_files_from_pr(
    repo_path: str, commit_sha: str, sync_mapping: list[tuple[str, ...]]
) -> bool:
    """
    Sync files from PR into target repo

    Args:
        repo_path (str): Path to repo.
        commit_sha (str): Commit SHA.
        sync_mapping (list[tuple[str, ...]]): Content of JSON file.

    Returns:
        bool: Mapping of source/target files from JSON.
    """
    has_changes = False

    for source_path, target_path in sync_mapping:
        target_dir = Path(target_path).parent
        if str(target_dir):
            run_mkdir(["-p", str(target_dir)], cwd=repo_path)

        stdout, _, return_code = run_git(
            ["show", f"{commit_sha}:{source_path}"],
            cwd=repo_path,
        )

        if return_code == 0 and stdout:
            full_target_path = Path(repo_path) / target_path
            full_target_path.parent.mkdir(parents=True, exist_ok=True)

            with open(full_target_path, "w", encoding="utf-8") as f:
                f.write(stdout)

            run_git(["add", target_path], cwd=repo_path)
            has_changes = True
        else:
            logger.warning(
                "Couldn't read file %s from commit %s",
                source_path,
                commit_sha[:8],
            )

    return has_changes


def commit_and_push_changes(commit_config: CommitConfig) -> None:
    """
    Commit and push changes

    Args:
        commit_config (CommitConfig): Schema of Commit.
    """
    if commit_config.json_changed and not commit_config.files_to_sync_found:
        commit_msg = (
            f"Update sync mapping from {commit_config.repo_name} " f"PR {commit_config.pr_number}"
        )
    else:
        commit_msg = f"Sync changes from {commit_config.repo_name} PR {commit_config.pr_number}"

    run_git(["commit", "-m", commit_msg], cwd=commit_config.repo_path)
    run_git(["push", "origin", commit_config.branch_name], cwd=commit_config.repo_path)


def create_or_update_pr(
    target_repo: str, branch_name: str, repo_name: str, pr_number: str, repo_path: str
) -> None:
    """
    Create or update PR in target repo

    Args:
        target_repo (str): Name of source repo.
        branch_name (str): Name of needed branch.
        repo_name (str): Name of target repo
        pr_number (str): Number of source PR.
        repo_path (str): Path to repo.
    """
    stdout, stderr, return_code = run_gh(
        [
            "pr",
            "list",
            "--repo",
            f"fipl-hse/{target_repo}",
            "--head",
            branch_name,
            "--json",
            "number",
        ]
    )

    target_pr_number = None
    if return_code == 0 and stdout:
        pr_list = json.loads(stdout) if stdout else []
        if pr_list and len(pr_list) > 0:
            target_pr_number = pr_list[0].get("number")

    run_git(["fetch", "origin", "main"], cwd=repo_path)

    stdout, stderr, return_code = run_git(
        ["log", "--oneline", f"origin/main..{branch_name}"], cwd=repo_path
    )

    has_commits = return_code == 0 and bool(stdout and stdout.strip())

    if has_commits:
        if target_pr_number is None:
            stdout, stderr, return_code = run_gh(
                [
                    "pr",
                    "create",
                    "--repo",
                    f"fipl-hse/{target_repo}",
                    "--head",
                    branch_name,
                    "--base",
                    "main",
                    "--title",
                    f"[Automated] Sync from {repo_name} PR {pr_number}",
                    "--body",
                    f"Automated synchronization from {repo_name} PR #{pr_number}",
                    "--label",
                    "automated pr",
                    "--assignee",
                    "demid5111",
                ]
            )

            if return_code == 1:
                logger.error("Failed to create PR. Exit code: %s", return_code)
                logger.error("stdout: %s", stdout)
                logger.error("stderr: %s", stderr)
                sys.exit(1)

            logger.info("Created new PR in target repository")

        else:
            stdout, stderr, return_code = run_gh(
                [
                    "pr",
                    "comment",
                    str(target_pr_number),
                    "--repo",
                    f"fipl-hse/{target_repo}",
                    "--body",
                    "Automatically updated",
                ]
            )

            if return_code != 0:
                logger.warning("Failed to update PR %s", target_pr_number)
    else:
        logger.info("No commits in branch %s - skipping PR creation", branch_name)


def validate_and_process_inputs() -> tuple[str, ...]:
    """
    Validating input args and processing basic information for script work

    Returns:
        tuple[str, ...]: Needed data from source repo
    """
    parser = SyncArgumentParser(underscores_to_dashes=True)
    args = parser.parse_args()

    repo_name = args.repo_name
    pr_number = args.pr_number
    target_repo = "fipl-hse.github.io"
    branch_name = f"auto-update-from-{repo_name}-pr-{pr_number}"
    root_dir = args.root_dir.resolve()
    toml_config = (args.toml_config_path or (root_dir / "pyproject.toml")).resolve()
    fileConfig(toml_config)

    gh_token = os.environ.get("GH_TOKEN")
    if not gh_token:
        logger.error("GH_TOKEN environment variable is not set")
        sys.exit(1)

    return repo_name, pr_number, target_repo, branch_name, gh_token


def prepare_target_repo(target_repo: str, branch_name: str, gh_token: str) -> None:
    """
    Prepare target repo for PR creation

    Args:
        target_repo (str): Name of target repo.
        branch_name (str): Name of branch in target repo.
        gh_token (str): Token to process operations.
    """
    clone_repo(target_repo, gh_token)
    setup_git_config(target_repo)
    check_and_create_label(target_repo)
    checkout_or_create_branch(branch_name, target_repo)


def get_pr_info(
    repo_name: str, pr_number: str, gh_token: str, target_repo: str
) -> tuple[str, list[str]]:
    """
    Get info about changes in PR from source repo

    Args:
        repo_name (str): Name of source repo.
        pr_number (str): Name of branch in source repo.
        gh_token (str): Token to process operations.
        target_repo (str): Name of target repo.

    Returns:
        tuple[str, list[str]]: Commit SHA and changed files.
    """
    pr_data = get_pr_data(repo_name, pr_number)

    if not pr_data:
        logger.error("PR data in source repo not found")
        sys.exit(0)

    commits = pr_data.get("commits", [])
    if not commits:
        logger.error("No commits found in PR %s", pr_number)
        sys.exit(0)

    commit_sha = commits[-1]["oid"] if commits else ""
    if not commit_sha:
        logger.error("Could not get commit SHA from PR %s", pr_number)
        sys.exit(0)

    changed_files = []

    if "files" in pr_data:
        changed_files = [f["path"] for f in pr_data["files"]]

    if not changed_files:
        logger.info("No changes found in PR %s", pr_number)
        sys.exit(0)

    add_remote_and_fetch(
        "parent-repo", f"https://{gh_token}@github.com/{repo_name}.git", target_repo
    )

    return commit_sha, changed_files


def run_sync(sync_config: SyncConfig) -> SyncResult:
    """
    Run final synchronization

    Args:
        sync_config (SyncConfig): Schema of Sync data.

    Returns:
        SyncResult: Result of sync.
    """
    has_changes = sync_config.json_changed
    files_to_sync_found = False

    sync_mapping = get_sync_mapping(sync_config.json_content) if sync_config.json_content else []

    sync_needed_files: list[tuple[str, ...]] = []
    for file in sync_config.changed_files:
        if file == TRACKED_JSON_PATH:
            continue

        for source, target in sync_mapping:
            if source == file:
                sync_needed_files.append((source, target))
                files_to_sync_found = True

    if sync_needed_files:
        has_synced = sync_files_from_pr(
            sync_config.target_repo, sync_config.commit_sha, sync_needed_files
        )
        has_changes = has_changes or has_synced

    return SyncResult(
        has_changes=has_changes,
        files_to_sync_found=files_to_sync_found,
        json_changed=sync_config.json_changed,
    )


def main() -> None:
    """
    Main function to create PR in target repo
    """
    repo_name, pr_number, target_repo, branch_name, gh_token = validate_and_process_inputs()

    prepare_target_repo(target_repo, branch_name, gh_token)

    commit_sha, changed_files = get_pr_info(repo_name, pr_number, gh_token, target_repo)

    json_content, json_changed = get_and_update_json_if_changed(
        repo_name, commit_sha, changed_files
    )

    sync_mapping = get_sync_mapping(json_content)
    has_files_to_sync = any(
        file != TRACKED_JSON_PATH and any(source == file for source, _ in sync_mapping)
        for file in changed_files
    )

    if not has_files_to_sync and not json_changed:
        logger.info("No files to sync and JSON not changed")
        sys.exit(0)

    sync_result = run_sync(
        SyncConfig(target_repo, changed_files, json_content, json_changed, commit_sha)
    )

    if sync_result.has_changes:
        commit_config = CommitConfig(
            target_repo,
            branch_name,
            repo_name,
            pr_number,
            sync_result.json_changed,
            sync_result.files_to_sync_found,
        )

        commit_and_push_changes(commit_config)
        create_or_update_pr(target_repo, branch_name, repo_name, pr_number, target_repo)
    else:
        logger.info("No changes to commit")
        sys.exit(0)


if __name__ == "__main__":
    main()
