"""
Domain-specific exceptions for the Lamb KB Server.

This module defines custom exception classes that represent domain-specific errors,
keeping them separate from HTTP or transport-specific error handling.
"""


class DomainException(Exception):
    """Base exception for all domain-specific errors."""
    pass


class ResourceNotFoundException(DomainException):
    """Raised when a requested resource cannot be found."""
    pass


class ValidationException(DomainException):
    """Raised when input validation fails."""
    pass


class ResourceAlreadyExistsException(DomainException):
    """Raised when trying to create a resource that already exists."""
    pass


class ConfigurationException(DomainException):
    """Raised when there's an issue with configuration parameters."""
    pass


class ProcessingException(DomainException):
    """Raised when processing a file or data fails."""
    pass


class AuthenticationException(DomainException):
    """Raised when authentication fails."""
    pass


class AuthorizationException(DomainException):
    """Raised when authorization fails."""
    pass


class DatabaseException(DomainException):
    """Raised when there's a database error."""
    pass


class PluginNotFoundException(DomainException):
    """Raised when a requested plugin cannot be found."""
    pass


class ExternalServiceException(DomainException):
    """Raised when an external service call fails."""
    pass


class FileNotFoundException(DomainException):
    """Raised when a file cannot be found."""
    pass


class InvalidParameterException(ValidationException):
    """Raised when a parameter is invalid."""
    pass