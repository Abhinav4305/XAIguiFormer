import torch
from data.EEGBenchmarkDataset import EEGBenchmarkDataset

class DebugDataset(EEGBenchmarkDataset):
    def process(self):
        try:
            super().process()
        except Exception as e:
            print(f"Exception caught during process/collate!")
            raise e

try:
    ds = DebugDataset('EEGBenchmarkDataset', 'BEED', 'train')
except Exception as e:
    import traceback
    traceback.print_exc()
