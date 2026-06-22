"""Namespace-agnostic XML traversal utilities for index and document parsing."""

from lxml import etree
from lxml.etree import (
    _Element,  # pyright: ignore[reportPrivateUsage]
)


def get_local_name(node: _Element) -> str:
    """Extracts the local tag name of an XML element, stripping any namespace.

    Args:
        node (_Element): The target XML element node.

    Returns:
        str: The un-namespaced local name component.
    """
    return etree.QName(node).localname


def find_child_by_local_name(parent: _Element, tag_name: str) -> _Element | None:
    """Finds the first child element matching a local tag name, bypassing namespaces.

    Args:
        parent (_Element): The parent element node to scan.
        tag_name (str): The local tag name target.

    Returns:
        _Element | None: The matching child instance if found, else None.
    """
    for child in parent:
        if etree.QName(child).localname == tag_name:
            return child

    return None


def find_children_by_local_name(parent: _Element, tag_name: str) -> list[_Element]:
    """Finds all child elements matching a local tag name, bypassing namespaces.

    Args:
        parent (_Element): The parent element node to scan.
        tag_name (str): The local tag name target.

    Returns:
        list[_Element]: A list of matching child element instances.
    """
    return [child for child in parent if etree.QName(child).localname == tag_name]


def get_child_text_by_local_name(parent: _Element, tag_name: str) -> str | None:
    """Extracts and cleans inner text content from a namespace-agnostic child element.

    Args:
        parent (_Element): The target parent element node.
        tag_name (str): The local tag name target.

    Returns:
        str | None:
            Cleaned inner string payload if the node exists and contains text,
            else None.
    """
    child = find_child_by_local_name(parent, tag_name)

    if child is not None and child.text:
        cleaned = child.text.strip()
        if cleaned:
            return cleaned

    return None
