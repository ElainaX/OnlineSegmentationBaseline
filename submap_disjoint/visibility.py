"""All-history visibility with compact rows and 63-bit packed frame words.

Only whether a positive raw mask touched a voxel is observable by MaskGraph.
The original raw mask value is always clamped to 0/1 before use. We preserve
that invariant, including zero overwrites, with no history truncation.
"""
import torch

class SparseVisibility:
    def __init__(self, max_voxels, frame_count, device):
        self.device = device
        self.frame_count = int(frame_count)
        self.word_count = (self.frame_count + 62) // 63
        self.row_index = torch.full((int(max_voxels),), -1, dtype=torch.int32, device=device)
        self.bits = torch.zeros((0, self.word_count), dtype=torch.int64, device=device)
        self.used = 0
        self.reads = 0
        self.peak_query_rows = 0
        self._shifts = torch.arange(63, dtype=torch.int64, device=device)

    def __setitem__(self, key, value):
        ids, frame = key
        frame = int(frame)
        if not 0 <= frame < self.frame_count:
            raise IndexError('segmentation frame outside visibility capacity')
        ids = torch.unique(ids.long().reshape(-1))
        if not ids.numel(): return
        rows = self.row_index[ids].long()
        missing = rows < 0
        missing_count = int(missing.sum())
        if missing_count:
            needed = self.used + missing_count
            if needed > len(self.bits):
                capacity = max(needed, max(4096, len(self.bits)*2))
                new = torch.zeros((capacity,self.word_count), dtype=torch.int64, device=self.device)
                new[:self.used] = self.bits[:self.used]
                self.bits = new
            new_rows = torch.arange(self.used,needed,device=self.device,dtype=torch.int64)
            self.row_index[ids[missing]] = new_rows.int()
            rows[missing] = new_rows
            self.used = needed
        bit = 1 << (frame % 63)
        word = frame // 63
        current = self.bits[rows,word]
        self.bits[rows,word] = (current | bit) if float(value)>0 else (current & ~bit)

    def __getitem__(self, ids):
        ids = ids.long().reshape(-1)
        rows = self.row_index[ids].long()
        self.reads += 1
        self.peak_query_rows = max(self.peak_query_rows, len(ids))
        out = torch.zeros((len(ids),self.frame_count),dtype=torch.float32,device=self.device)
        if self.used == 0 or len(ids) == 0: return out
        valid = rows >= 0
        words = self.bits[rows.clamp_min(0)]
        words = torch.where(valid[:,None],words,torch.zeros_like(words))
        for w in range(self.word_count):
            start = w*63; n = min(63,self.frame_count-start)
            out[:,start:start+n] = ((words[:,w,None] >> self._shifts[:n]) & 1).float()
        return out

    def stats(self):
        return dict(observed_voxels=self.used, frame_capacity=self.frame_count,
                    allocated_bytes=self.bits.numel()*8+self.row_index.numel()*4,
                    original_dense_bytes=self.row_index.numel()*self.frame_count*4,
                    reads=self.reads, peak_query_rows=self.peak_query_rows,
                    history_truncated=False)
