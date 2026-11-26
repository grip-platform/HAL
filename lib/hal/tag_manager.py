"""HAL Tag Manager — categorise and label conversations and resources.

Provides tagging for conversations, memory entries, and resources.
Enables filtered retrieval and organisation.

Usage:
    from lib.hal.tag_manager import TagManager
    tags = TagManager()
    tags.add("session-123", "bug-fix")
    tags.add("session-123", "authentication")
    results = tags.find_by_tag("bug-fix")
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TagInfo:
    tag: str
    count: int


class TagManager:
    """Manages tags for resources."""

    def __init__(self):
        self._resource_tags: dict[str, set[str]] = {}
        self._tag_resources: dict[str, set[str]] = {}

    def add(self, resource_id: str, tag: str) -> None:
        if resource_id not in self._resource_tags:
            self._resource_tags[resource_id] = set()
        self._resource_tags[resource_id].add(tag)
        if tag not in self._tag_resources:
            self._tag_resources[tag] = set()
        self._tag_resources[tag].add(resource_id)

    def remove(self, resource_id: str, tag: str) -> bool:
        removed = False
        if resource_id in self._resource_tags:
            self._resource_tags[resource_id].discard(tag)
            removed = True
        if tag in self._tag_resources:
            self._tag_resources[tag].discard(resource_id)
        return removed

    def get_tags(self, resource_id: str) -> set[str]:
        return set(self._resource_tags.get(resource_id, set()))

    def find_by_tag(self, tag: str) -> set[str]:
        return set(self._tag_resources.get(tag, set()))

    def all_tags(self) -> list[TagInfo]:
        return sorted(
            [TagInfo(tag, len(resources)) for tag, resources in self._tag_resources.items()],
            key=lambda t: -t.count,
        )

    def clear(self, resource_id: str) -> None:
        tags = self._resource_tags.pop(resource_id, set())
        for tag in tags:
            if tag in self._tag_resources:
                self._tag_resources[tag].discard(resource_id)
