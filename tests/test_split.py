import unittest

from main import grouped_holdout_indices


class GroupedHoldoutTests(unittest.TestCase):
    def test_exact_messages_never_cross_partitions(self):
        messages = [f"ham message {i}" for i in range(80)] + [f"spam message {i}" for i in range(80)]
        labels = ["ham"] * 80 + ["spam"] * 80
        # Replicate some strings so the test proves group integrity, not just row splitting.
        messages += messages[:20]
        labels += labels[:20]
        train, test = grouped_holdout_indices(messages, labels)
        self.assertFalse(set(messages[i] for i in train) & set(messages[i] for i in test))
        self.assertTrue(set(labels[i] for i in train) == {"ham", "spam"})
        self.assertTrue(set(labels[i] for i in test) == {"ham", "spam"})

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            grouped_holdout_indices(["one"], ["ham", "spam"])


if __name__ == "__main__":
    unittest.main()
