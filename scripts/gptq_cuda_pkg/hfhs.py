"""Hipfire Hessian Sidecar (HFHS v1) reader.

Reads the binary format produced by `scripts/collect_hessian.py` and
consumed in Rust by `crates/hipfire-quantize/src/hessian_io.rs`. The
canonical format spec is `docs/plans/gptq-hessian-format.md` §3.

Wire format (little-endian throughout):
  Header (24 bytes):
    [0..4)   magic     b"HFHS"
    [4..8)   version   u32 = 1
    [8..16)  n_tensors u64
    [16..24) reserved  u64 = 0
  Per-record:
    [0..4)            name_len     u32
    [4..4+name_len)   name         utf-8 bytes
    [+4..)            expert_idx   u32 (0 for non-MoE)
    [+4..)            K            u32
    [+4..)            dtype_flag   u32 (1=F32, 2=F64)
    [+K*K*sz..)       payload      row-major K×K dtype values

The 9B sidecar is 33 GB on disk — we mmap it and slice per-tensor
on demand to avoid blowing the 32 GB system RAM budget (plan §4.4).
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


HFHS_MAGIC = b"HFHS"
HFHS_VERSION = 1
HEADER_SIZE = 24
DTYPE_F32 = 1
DTYPE_F64 = 2


@dataclass
class HessianEntry:
    name: str
    expert_idx: int
    k: int
    dtype: str             # "f32" or "f64"
    payload_offset: int
    payload_bytes: int


class HessianSidecar:
    """Lazy-loading reader for an HFHS v1 sidecar.

    `open()` parses the header + record index in one pass; per-tensor
    payloads stay on disk until `get(name)` slices them.
    """

    def __init__(self, path: Path):
        self.path = path
        self._fd = open(path, "rb")
        self._mmap = mmap.mmap(self._fd.fileno(), 0, prot=mmap.PROT_READ)
        try:
            mmap.MADV_SEQUENTIAL  # type: ignore
            self._mmap.madvise(mmap.MADV_SEQUENTIAL)  # type: ignore
        except (AttributeError, OSError):
            pass

        if len(self._mmap) < HEADER_SIZE:
            raise IOError(f"HFHS truncated: {len(self._mmap)} bytes < header {HEADER_SIZE}")
        magic = bytes(self._mmap[0:4])
        if magic != HFHS_MAGIC:
            raise IOError(f"invalid HFHS magic: {magic!r}, expected {HFHS_MAGIC!r}")
        version = struct.unpack_from("<I", self._mmap, 4)[0]
        if version != HFHS_VERSION:
            raise IOError(f"unsupported HFHS version {version}, expected {HFHS_VERSION}")
        n_tensors = struct.unpack_from("<Q", self._mmap, 8)[0]

        index: dict[tuple[str, int], HessianEntry] = {}
        pos = HEADER_SIZE
        for _ in range(n_tensors):
            (name_len,) = struct.unpack_from("<I", self._mmap, pos); pos += 4
            name = self._mmap[pos:pos + name_len].decode("utf-8")
            pos += name_len
            (expert_idx,) = struct.unpack_from("<I", self._mmap, pos); pos += 4
            (k,) = struct.unpack_from("<I", self._mmap, pos); pos += 4
            (dtype_flag,) = struct.unpack_from("<I", self._mmap, pos); pos += 4
            if dtype_flag == DTYPE_F32:
                dtype = "f32"
                elem_size = 4
            elif dtype_flag == DTYPE_F64:
                dtype = "f64"
                elem_size = 8
            else:
                raise IOError(f"unknown HFHS dtype flag {dtype_flag} for tensor {name!r}")
            payload_bytes = k * k * elem_size
            index[(name, expert_idx)] = HessianEntry(
                name=name,
                expert_idx=expert_idx,
                k=k,
                dtype=dtype,
                payload_offset=pos,
                payload_bytes=payload_bytes,
            )
            pos += payload_bytes
        if pos > len(self._mmap):
            raise IOError(f"HFHS truncated at offset {pos}, file is {len(self._mmap)} bytes")

        self.index = index

    def __enter__(self) -> "HessianSidecar":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._mmap.close()
        finally:
            self._fd.close()

    def has(self, name: str, expert_idx: int = 0) -> bool:
        return (name, expert_idx) in self.index

    def get(self, name: str, expert_idx: int = 0) -> HessianEntry:
        """Returns the entry, raising KeyError if missing."""
        return self.index[(name, expert_idx)]

    def names(self) -> list[str]:
        """All registered base names (no expert_idx). Sorted for
        deterministic iteration order across runs."""
        return sorted({name for (name, _) in self.index.keys()})

    def load_f64(
        self,
        name: str,
        expert_idx: int = 0,
        *,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Load one Hessian as a contiguous FP64 torch tensor, K×K, on
        the requested device. Cast happens during the load (numpy ←
        mmap slice, then torch from_numpy, then `.to(float64)`); we do
        not keep the mmap-backed FP32 around longer than needed.
        """
        e = self.get(name, expert_idx)
        np_dtype = np.float32 if e.dtype == "f32" else np.float64
        buf = self._mmap[e.payload_offset:e.payload_offset + e.payload_bytes]
        arr = np.frombuffer(buf, dtype=np_dtype).reshape(e.k, e.k)
        # Copy off the mmap so the returned tensor doesn't reference
        # the file mapping (which we may unmap before the caller is done).
        arr_owned = np.array(arr, copy=True)
        t = torch.from_numpy(arr_owned).to(torch.float64)
        if device != "cpu":
            t = t.to(device, non_blocking=True)
        return t
