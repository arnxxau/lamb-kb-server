"""
Exception hierarchy for the Lamb KB Server.

*No* transport-level (FastAPI) classes are imported here – keep those concerns
in the presentation layer.
"""

# ───────────────────────────── base classes ────────────────────────────── #

class AppException(Exception):
    """Root of **all** application errors."""
    pass


# ─────────────────────────── domain / business ────────────────────────── #

class DomainException(AppException):
    """Errors that make sense inside the domain language."""
    pass


class ResourceNotFoundException(DomainException):
    """The requested entity does not exist."""
    pass


class ResourceAlreadyExistsException(DomainException):
    """A unique key is duplicated (conflict)."""
    pass


class ValidationException(DomainException):
    """Request payload / parameters do not satisfy domain constraints."""
    pass


class ConfigurationException(DomainException):
    """Server-side configuration is invalid or incomplete."""
    pass


class ProcessingException(DomainException):
    """Generic failure while applying business logic."""
    pass


class AuthenticationException(DomainException):   # optional, keep if used
    pass


class AuthorizationException(DomainException):    # optional, keep if used
    pass


# ──────────────────────── repository / infrastructure ─────────────────── #

class RepositoryException(AppException):
    """Base for data-access and infrastructure errors."""
    pass


class DataIntegrityException(RepositoryException):
    """Constraint violation (unique, FK, …)."""
    pass


class DatabaseException(RepositoryException):
    """Unexpected DB error that is *not* a constraint violation."""
    pass


class ExternalServiceException(RepositoryException):
    """A call to a third-party service failed (network, SDK…)."""
    pass


# ────────────────────────── aliases (optional) ────────────────────────── #
# These preserve backwards compatibility with names you already used.

FileNotFoundException       = ResourceNotFoundException
PluginNotFoundException     = ResourceNotFoundException
InvalidParameterException   = ValidationException