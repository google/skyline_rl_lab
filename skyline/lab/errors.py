"""Errors related to Skyline project."""


class EnvError(Exception):
  """A base class for errors related to Skyline environment."""


class IllegalActionError(EnvError):
  """Illegal action being conducted."""


class UnknownLabEnvError(EnvError):
  """Unknown lab environment."""
