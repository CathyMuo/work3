import unittest

from work3.wiki_rsi_decode import native_gpu_ids


class WikiDecodeDevicesTest(unittest.TestCase):
    def test_selects_distinct_additional_devices(self):
        self.assertEqual(native_gpu_ids("0", None), [])
        self.assertEqual(native_gpu_ids("0", "1, 2,3"), ["1", "2", "3"])
        self.assertEqual(native_gpu_ids("0", "2"), ["2"])

    def test_rejects_reusing_a_physical_gpu(self):
        for legacy, native in (("0", "1,01"), ("0", "0,2"), ("0,1", "2"), (None, "2"),
                               ("0", "1,2,3,4"), ("0", "1,")):
            with self.subTest(legacy=legacy, native=native), self.assertRaises(ValueError):
                native_gpu_ids(legacy, native)
