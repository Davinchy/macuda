import unittest

from read_boot_log import pc_relative_references, records, transcript


class BootLogTests(unittest.TestCase):
    def test_pc_relative_metadata_and_signed_upper(self):
        text = """ 1004e24: 1f000617 auipc a2, 0x1f000
 1004e28: 9b460613 addi a2, a2, -0x64c
 1006af6: ffffc097 auipc ra, 0xffffc
 1006afa: 82e080e7 addi ra, ra, -0x7d2
"""
        self.assertEqual(pc_relative_references(text, {0x200047d8, 0x1002324}),
                         [{"pc": "0x1004e24", "target": "0x200047d8"},
                          {"pc": "0x1006af6", "target": "0x1002324"}])

    def test_arguments_precede_metadata(self):
        words = {0: 9, 1: 1, 2: 0x02030000000047d8, 3: 100,
                 4: 1, 5: 2, 6: 0, 7: 0x1006b0c,
                 8: 0x0106000000001ac0, 9: 200}
        decoded = records(words)
        self.assertEqual(decoded[0]["args"], ["0x1"])
        self.assertEqual(decoded[1]["args"], ["0x1", "0x2", "0x0", "0x1006b0c"])

    def test_missing_or_capped_data_refused(self):
        for words in ({0: 96}, {0: 3, 2: 0x02030000000047d8, 3: 100},
                      {0: 3, 1: 1, 2: 0, 3: 100}):
            with self.assertRaises(ValueError):
                records(words)

    def test_empty_log_and_zero_argument(self):
        text = """tinynv: gsp log LOGINTR: empty
tinynv: gsp log LOGINIT: 4 non-zero u64 words:
  [00] 0000000000000003
  [01] 0000000000000000
  [02] 0203000000004bb0
  [03] 0000000000000009
"""
        self.assertEqual(transcript(text)[0]["records"][0]["args"], ["0x0"])


if __name__ == "__main__":
    unittest.main()
