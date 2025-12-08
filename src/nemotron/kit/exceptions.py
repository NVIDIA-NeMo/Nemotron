"""
Exception classes for the nemotron.art package.
"""


class ArtifactNotFoundError(Exception):
    """Raised when an artifact cannot be found in the registry."""

    def __init__(self, name: str, message: str | None = None) -> None:
        self.name = name
        self.message = message or f"Artifact not found: {name}"
        super().__init__(self.message)


class ArtifactVersionNotFoundError(Exception):
    """Raised when a specific version of an artifact cannot be found."""

    def __init__(
        self, name: str, version: str | int, message: str | None = None
    ) -> None:
        self.name = name
        self.version = version
        self.message = message or f"Artifact version not found: {name}:{version}"
        super().__init__(self.message)
