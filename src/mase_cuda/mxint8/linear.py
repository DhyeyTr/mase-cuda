import math
import torch

from .quantize import quantize1d_simulated
from .dequantize import dequantize1d, dequantize1d_simulated


class PackedWeight:
    def __init__(self, shape: tuple[int], group_size: int, device=None, dtype=torch.bfloat16):
        numel = math.prod(shape)
        assert numel % group_size == 0, "Number of elements in the weights tensor must be divisible by the group size"
        self.shape = shape
        self.numel = numel
        self.group_size = group_size
        self.dtype = dtype

        self.weight = torch.empty(self.numel, device=device, dtype=torch.uint8)
        self.scales = torch.empty(self.numel // group_size, device=device, dtype=torch.uint8)

    def unpack(self) -> torch.Tensor:
        try:
            return self.unpack_accelerated()
        except NotImplementedError:
            return self.unpack_simulated()

    def unpack_accelerated(self) -> torch.Tensor:
        if not self.weight.is_contiguous():
            self.weight = self.weight.contiguous()

        w = dequantize1d(self.weight, self.scales, self.group_size).reshape(self.shape).to(self.dtype)
        return w

    def unpack_simulated(self) -> torch.Tensor:
        if not self.weight.is_contiguous():
            self.weight = self.weight.contiguous()

        w = dequantize1d_simulated(self.weight, self.scales, self.group_size).reshape(self.shape).to(self.dtype)
        return w

    @classmethod
    def pack(cls, weights: torch.Tensor, group_size: int) -> "PackedWeight":
        try:
            return cls.pack_accelerated(weights, group_size)
        except NotImplementedError:
            return cls.pack_simulated(weights, group_size)

    @classmethod
    def pack_accelerated(cls, weights: torch.Tensor, group_size: int) -> "PackedWeight":
        raise NotImplementedError

    @classmethod
    def pack_simulated(cls, weights: torch.Tensor, group_size: int) -> "PackedWeight":
        weights = weights.contiguous()
        device = weights.device
        ori_shape = weights.size()
        ori_dtype = weights.dtype
        weights = weights.flatten()
        w, s = quantize1d_simulated(weights, group_size)
        packed = cls(ori_shape, group_size, device, ori_dtype)
        packed.weight = w
        packed.scales = s
        return packed

    @property
    def nbytes(self) -> float:
        return self.weight.nbytes + self.scales.nbytes


class QLinearPacked(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=torch.bfloat16,
        group_size: int = 16,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.packed_weight = PackedWeight((out_features, in_features), group_size, device=device, dtype=dtype)
        if bias:
            self.packed_bias = PackedWeight((out_features,), group_size, device=device, dtype=dtype)
        else:
            self.packed_bias = None
        
        self.chunk_size = chunk_size
    
    def _unpack_weight_rows(self, start: int, end: int) -> torch.Tensor:
        rows = end - start
        cols = self.in_features

        flat_start = start * cols
        flat_end = end * cols

        # Slice quantized weights
        weight_q = self.packed_weight.weight[flat_start:flat_end]

        # Slice scales. Since each scale covers group_size elements in the flattened layout,
        # these indices must align to group boundaries.
        assert flat_start % self.packed_weight.group_size == 0
        assert flat_end % self.packed_weight.group_size == 0

        scale_start = flat_start // self.packed_weight.group_size
        scale_end = flat_end // self.packed_weight.group_size
        scales_q = self.packed_weight.scales[scale_start:scale_end]

        try:
            w = dequantize1d(weight_q, scales_q, self.packed_weight.group_size)
        except NotImplementedError:
            w = dequantize1d_simulated(weight_q, scales_q, self.packed_weight.group_size)

        return w.reshape(rows, cols).to(self.packed_weight.dtype)

    def _unpack_bias_rows(self, start: int, end: int) -> torch.Tensor:
        if self.packed_bias is None:
            return torch.zeros(end - start, device=self.packed_weight.weight.device, dtype=self.packed_weight.dtype)

        # Bias is 1D, so we can slice directly
        bias_q = self.packed_bias.weight[start:end]

        # Each bias element corresponds to one output feature, so each has its own scale
        scales_q = self.packed_bias.scales[start:end]

        try:
            b = dequantize1d(bias_q, scales_q, self.packed_bias.group_size)
        except NotImplementedError:
            b = dequantize1d_simulated(bias_q, scales_q, self.packed_bias.group_size)

        return b.to(self.packed_bias.dtype)

    @torch.no_grad()
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Old behavior: dequantize the whole weight matrix
        if self.chunk_size is None or self.chunk_size >= self.out_features:
            return F.linear(
                input,
                self.packed_weight.unpack(),
                self.packed_bias.unpack() if self.packed_bias is not None else None,
            )

        # Chunked behavior
        chunk_size = self.chunk_size

        # To keep scale indexing simple and correct, chunk boundaries must align with group_size.
        if chunk_size % self.packed_weight.group_size != 0:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be divisible by weight group_size "
                f"({self.packed_weight.group_size})"
            )

        if self.packed_bias is not None and chunk_size % self.packed_bias.group_size != 0:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be divisible by bias group_size "
                f"({self.packed_bias.group_size})"
            )

        outputs = []
        for start in range(0, self.out_features, chunk_size):
            end = min(start + chunk_size, self.out_features)

            # To make the last chunk safe too, require out_features to align.
            # Or you can pad during packing if needed.
            if end - start != chunk_size and end != self.out_features:
                raise RuntimeError("Unexpected chunk sizing error")

            w_chunk = self._unpack_weight_rows(start, end)
            b_chunk = self._unpack_bias_rows(start, end)

            out_chunk = torch.nn.functional.linear(input, w_chunk, b_chunk)
            outputs.append(out_chunk)

        return torch.cat(outputs, dim=-1)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.packed_bias is not None}, group_size={self.packed_weight.group_size}, nbits={self.nbits}, chunk_size={self.chunk_size}"

    @property
    def nbits(self) -> float:
        nbytes = self.packed_weight.nbytes
        if self.packed_bias is not None:
            nbytes += self.packed_bias.nbytes

        numels = self.in_features * self.out_features
        if self.packed_bias is not None:
            numels += self.out_features
        return nbytes * 8 / numels

    @classmethod
    def build_from_linear(cls, linear: torch.nn.Linear, group_size: int = 16, chunk_size: int = None) -> "QLinearPacked":
        device = linear.weight.device
        qlinear = cls(
            linear.in_features, linear.out_features, bias=linear.bias is not None, device=device, group_size=group_size, chunk_size=chunk_size
        )
        qlinear.packed_weight = PackedWeight.pack_simulated(linear.weight, group_size)
        if linear.bias is not None:
            qlinear.packed_bias = PackedWeight.pack_simulated(linear.bias, group_size)
        return qlinear
