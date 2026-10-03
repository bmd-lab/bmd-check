from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tomllib
from typing import Any, Mapping


CONFIG_ENV_VAR = "BMD_AGENT_RESOURCES"
DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_REMOTE_COMMAND_TIMEOUT_SECONDS = 20
DEFAULT_SCHEDULER_ACCOUNTING_TIMEOUT_SECONDS = 60

_PARTITION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SSH_HOST_RE = re.compile(r"^[A-Za-z0-9_.@:-]+$")
_DEPLOYMENT_PROFILE_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


class ConfigurationError(RuntimeError):
    """Raised when resource configuration is missing or invalid."""


@dataclass(frozen=True)
class GitRepositoryResource:
    key: str
    name: str
    path: Path
    role: str
    access: str
    protected: bool
    live: bool
    capability_python: Path | None = None


@dataclass(frozen=True)
class SlurmClusterResource:
    key: str
    name: str
    ssh_host: str
    partition: str
    access: str
    allowed_remote_roots: tuple[PurePosixPath, ...]
    deployment_profile: str | None = None
    ssh_connect_timeout_seconds: int = DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS
    remote_command_timeout_seconds: int = DEFAULT_REMOTE_COMMAND_TIMEOUT_SECONDS
    scheduler_accounting_timeout_seconds: int = DEFAULT_SCHEDULER_ACCOUNTING_TIMEOUT_SECONDS


@dataclass(frozen=True)
class ResourceRegistry:
    repositories: dict[str, GitRepositoryResource]
    clusters: dict[str, SlurmClusterResource]


def project_root() -> Path:
    """Return the root directory of the bmd_agent source checkout."""
    return Path(__file__).resolve().parents[2]


def resources_example_path() -> Path:
    """Return the tracked example resource configuration path."""
    return project_root() / "config" / "resources.example.toml"


def user_config_path(
    *,
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Return the conventional per-user resource configuration path."""

    platform_name = platform_name or sys.platform
    environ = environ or os.environ
    home = home or Path.home()

    if platform_name == "win32":
        base = environ.get("APPDATA")
        config_home = Path(base) if base else home / "AppData" / "Roaming"
        return config_home / "bmd-agent" / "resources.toml"

    if platform_name == "darwin":
        return home / "Library" / "Application Support" / "bmd-agent" / "resources.toml"

    base = environ.get("XDG_CONFIG_HOME")
    config_home = Path(base) if base else home / ".config"
    return config_home / "bmd-agent" / "resources.toml"


def resolve_resources_path(
    *,
    environ: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    home: Path | None = None,
) -> Path:
    """Resolve the deployment-local resource configuration path."""

    environ = environ or os.environ
    configured = environ.get(CONFIG_ENV_VAR)

    if configured:
        return Path(configured).expanduser()

    return user_config_path(
        platform_name=platform_name,
        environ=environ,
        home=home,
    )


def load_resources(path: Path | str | None = None) -> ResourceRegistry:
    """Load and validate configured BMD resources."""

    config_path = Path(path).expanduser() if path is not None else resolve_resources_path()

    if not config_path.exists():
        raise ConfigurationError(_missing_config_message(config_path))

    with config_path.open("rb") as handle:
        raw_config = tomllib.load(handle)

    return parse_resources(raw_config, source=config_path)


def parse_resources(config: Mapping[str, Any], *, source: Path | str = "<memory>") -> ResourceRegistry:
    """Validate resource configuration loaded from TOML."""

    repositories = {
        key: _parse_repository(key, value, source=source)
        for key, value in _table(config, "repositories", source=source).items()
    }

    clusters = {
        key: _parse_cluster(key, value, source=source)
        for key, value in _table(config, "clusters", source=source).items()
    }

    return ResourceRegistry(repositories=repositories, clusters=clusters)


def _parse_repository(
    key: str,
    value: Any,
    *,
    source: Path | str,
) -> GitRepositoryResource:
    table = _mapping(value, f"repositories.{key}", source=source)
    access = _required_str(table, "access", f"repositories.{key}", source=source)

    if access != "read_only":
        raise ConfigurationError(
            f"{source}: repositories.{key}.access must be 'read_only', got {access!r}"
        )

    return GitRepositoryResource(
        key=key,
        name=_required_str(table, "name", f"repositories.{key}", source=source),
        path=Path(_required_str(table, "path", f"repositories.{key}", source=source)).expanduser(),
        role=_required_str(table, "role", f"repositories.{key}", source=source),
        access=access,
        protected=_required_bool(table, "protected", f"repositories.{key}", source=source),
        live=_required_bool(table, "live", f"repositories.{key}", source=source),
        capability_python=_optional_path(table, "capability_python", f"repositories.{key}", source=source),
    )


def _parse_cluster(
    key: str,
    value: Any,
    *,
    source: Path | str,
) -> SlurmClusterResource:
    table = _mapping(value, f"clusters.{key}", source=source)
    access = _required_str(table, "access", f"clusters.{key}", source=source)

    if access != "observational":
        raise ConfigurationError(
            f"{source}: clusters.{key}.access must be 'observational', got {access!r}"
        )

    partition = _required_str(table, "partition", f"clusters.{key}", source=source)

    if not _PARTITION_RE.fullmatch(partition):
        raise ConfigurationError(
            f"{source}: clusters.{key}.partition contains unsafe characters"
        )

    ssh_host = _required_str(table, "ssh_host", f"clusters.{key}", source=source)

    if not _SSH_HOST_RE.fullmatch(ssh_host):
        raise ConfigurationError(
            f"{source}: clusters.{key}.ssh_host contains unsafe characters"
        )

    roots = _required_list(table, "allowed_remote_roots", f"clusters.{key}", source=source)
    allowed_remote_roots = tuple(
        _parse_remote_root(root, f"clusters.{key}.allowed_remote_roots", source=source)
        for root in roots
    )

    if not allowed_remote_roots:
        raise ConfigurationError(
            f"{source}: clusters.{key}.allowed_remote_roots must not be empty"
        )

    deployment_profile = _optional_str(
        table,
        "deployment_profile",
        f"clusters.{key}",
        source=source,
    )
    if deployment_profile is not None and not _DEPLOYMENT_PROFILE_RE.fullmatch(
        deployment_profile
    ):
        raise ConfigurationError(
            f"{source}: clusters.{key}.deployment_profile is invalid"
        )

    return SlurmClusterResource(
        key=key,
        name=_required_str(table, "name", f"clusters.{key}", source=source),
        ssh_host=ssh_host,
        partition=partition,
        access=access,
        allowed_remote_roots=allowed_remote_roots,
        deployment_profile=deployment_profile,
        ssh_connect_timeout_seconds=_optional_positive_int(
            table,
            "ssh_connect_timeout_seconds",
            f"clusters.{key}",
            default=DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS,
            source=source,
        ),
        remote_command_timeout_seconds=_optional_positive_int(
            table,
            "remote_command_timeout_seconds",
            f"clusters.{key}",
            default=DEFAULT_REMOTE_COMMAND_TIMEOUT_SECONDS,
            source=source,
        ),
        scheduler_accounting_timeout_seconds=_optional_positive_int(
            table,
            "scheduler_accounting_timeout_seconds",
            f"clusters.{key}",
            default=DEFAULT_SCHEDULER_ACCOUNTING_TIMEOUT_SECONDS,
            source=source,
        ),
    )


def _parse_remote_root(value: Any, field: str, *, source: Path | str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{source}: {field} entries must be non-empty strings")

    root = PurePosixPath(value)

    if not root.is_absolute():
        raise ConfigurationError(f"{source}: {field} entries must be absolute POSIX paths")

    if ".." in root.parts:
        raise ConfigurationError(f"{source}: {field} entries must not contain '..'")

    return root


def _table(config: Mapping[str, Any], field: str, *, source: Path | str) -> Mapping[str, Any]:
    value = config.get(field)

    if value is None:
        raise ConfigurationError(f"{source}: missing [{field}] table")

    return _mapping(value, field, source=source)


def _mapping(value: Any, field: str, *, source: Path | str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{source}: {field} must be a table")

    return value


def _required_str(
    table: Mapping[str, Any],
    key: str,
    field: str,
    *,
    source: Path | str,
) -> str:
    value = table.get(key)

    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{source}: {field}.{key} must be a non-empty string")

    return value


def _required_bool(
    table: Mapping[str, Any],
    key: str,
    field: str,
    *,
    source: Path | str,
) -> bool:
    value = table.get(key)

    if not isinstance(value, bool):
        raise ConfigurationError(f"{source}: {field}.{key} must be true or false")

    return value


def _required_list(
    table: Mapping[str, Any],
    key: str,
    field: str,
    *,
    source: Path | str,
) -> list[Any]:
    value = table.get(key)

    if not isinstance(value, list):
        raise ConfigurationError(f"{source}: {field}.{key} must be a list")

    return value


def _optional_path(
    table: Mapping[str, Any],
    key: str,
    field: str,
    *,
    source: Path | str,
) -> Path | None:
    value = table.get(key)

    if value is None:
        return None

    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{source}: {field}.{key} must be a non-empty string")

    return Path(value).expanduser()


def _optional_str(
    table: Mapping[str, Any],
    key: str,
    field: str,
    *,
    source: Path | str,
) -> str | None:
    value = table.get(key)

    if value is None:
        return None

    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{source}: {field}.{key} must be a non-empty string")

    return value


def _optional_positive_int(
    table: Mapping[str, Any],
    key: str,
    field: str,
    *,
    default: int,
    source: Path | str,
) -> int:
    value = table.get(key, default)

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigurationError(f"{source}: {field}.{key} must be a positive integer")

    return value


def _missing_config_message(path: Path) -> str:
    example_path = resources_example_path()
    if example_path.is_file():
        # Running from a source checkout: the tracked example is right here.
        example = f"Create one from {example_path}"
    else:
        # Installed package: the example lives in the repository, not the package.
        example = (
            "Create one from config/resources.example.toml in the BMD Agent repository"
        )

    return (
        f"No BMD Agent resource configuration found at {path}. "
        f"{example}, or set {CONFIG_ENV_VAR} to a deployment-local resources.toml file "
        "(a shared installation's launcher normally sets it). "
        "The example configuration is not used for real execution."
    )
