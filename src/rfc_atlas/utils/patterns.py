"""Design pattern primitives for the RFC Atlas."""

from typing import ClassVar

from rfc_atlas.utils.exceptions import SingletonViolationError


class StaticSingleton:
    """Base class for static-only registries to enforce singleton mechanics."""

    is_instantiated: ClassVar[bool] = False

    def __init__(self) -> None:
        """Enforces strict singleton instantiation.

        Raises:
            SingletonViolationError: If an attempt is made to instantiate a
                strict Singleton twice.
        """
        if self.__class__.is_instantiated:
            raise SingletonViolationError(self.__class__.__name__)
        self.__class__.is_instantiated = True
