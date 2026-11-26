"""Tests for HAL tag manager."""
import os, sys, unittest
from lib.hal.tag_manager import TagManager

class TestTagManager(unittest.TestCase):
    def test_add_and_get(self):
        t = TagManager()
        t.add("r1", "bug")
        self.assertIn("bug", t.get_tags("r1"))

    def test_find_by_tag(self):
        t = TagManager()
        t.add("r1", "bug")
        t.add("r2", "bug")
        self.assertEqual(len(t.find_by_tag("bug")), 2)

    def test_remove(self):
        t = TagManager()
        t.add("r1", "bug")
        t.remove("r1", "bug")
        self.assertNotIn("bug", t.get_tags("r1"))

    def test_all_tags(self):
        t = TagManager()
        t.add("r1", "a")
        t.add("r2", "a")
        t.add("r3", "b")
        tags = t.all_tags()
        self.assertEqual(tags[0].tag, "a")
        self.assertEqual(tags[0].count, 2)

    def test_clear(self):
        t = TagManager()
        t.add("r1", "a")
        t.add("r1", "b")
        t.clear("r1")
        self.assertEqual(len(t.get_tags("r1")), 0)

    def test_empty(self):
        self.assertEqual(len(TagManager().get_tags("x")), 0)

if __name__ == "__main__":
    unittest.main()
