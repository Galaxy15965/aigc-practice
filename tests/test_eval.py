import unittest

from eval_agent import build_summary, is_refusal, select_questions


class EvalTests(unittest.TestCase):
    def test_stratified_selection_includes_unanswerable(self):
        questions = [
            {"q": f"answerable-{index}", "page": index}
            for index in range(1, 11)
        ] + [
            {"q": f"unanswerable-{index}", "page": None}
            for index in range(1, 6)
        ]

        selected = select_questions(questions, 4, "stratified")

        self.assertEqual(len(selected), 4)
        self.assertEqual(sum(item["page"] is None for item in selected), 1)
        self.assertEqual(sum(item["page"] is not None for item in selected), 3)

    def test_limit_zero_selects_all_questions(self):
        questions = [{"q": "a", "page": 1}, {"q": "b", "page": None}]
        self.assertEqual(select_questions(questions, 0), questions)

    def test_refusal_markers(self):
        self.assertTrue(is_refusal("文档中没有提到这个问题。"))
        self.assertTrue(is_refusal("资料中没有相关内容。"))
        self.assertFalse(is_refusal("依据：第 11 页。"))

    def test_summary_counts_answer_and_refusal(self):
        records = [
            {
                "expected_page": 11,
                "success": True,
                "refused": False,
                "retrieval_hit": True,
                "citation_hit": True,
                "citation_valid": True,
                "tool_calls": 2,
                "elapsed_s": 2.0,
                "trace": [{"ok": True}, {"ok": True}],
            },
            {
                "expected_page": None,
                "success": True,
                "refused": True,
                "retrieval_hit": None,
                "citation_hit": None,
                "citation_valid": None,
                "tool_calls": 1,
                "elapsed_s": 4.0,
                "trace": [{"ok": True}],
            },
        ]

        summary = build_summary(records)

        self.assertEqual(summary["successes"], 2)
        self.assertEqual(summary["retrieval_hit_rate"], 1.0)
        self.assertEqual(summary["refusal_accuracy"], 1.0)
        self.assertEqual(summary["tool_call_success_rate"], 1.0)
        self.assertEqual(summary["avg_elapsed_s"], 3.0)


if __name__ == "__main__":
    unittest.main()
