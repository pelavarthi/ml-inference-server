"""Quantization: shrink a model's weights to fewer bits per weight.

A full-precision weight is a 64-bit float. Most models do not need
that much precision, so each weight can be rounded to the nearest step
on a small grid instead: 8 bits gives 256 steps, 4 bits gives 16.
Storing the grid position (the code) plus one scale and one offset
per model is far smaller than storing every float, and the prediction
only shifts by at most half a grid step per weight.

This is affine quantization: code = round((weight - min) / scale),
where scale = (max - min) / (levels - 1). The only information lost
is the rounding, so the error is bounded and easy to measure.
"""

import math

SUPPORTED_BITS = (4, 8, 16)
BYTES_PER_FLOAT = 8


class QuantizedWeights:
    """A weight vector stored as small integer codes plus a scale."""

    def __init__(self, codes, scale, zero_point, bits):
        self.codes = list(codes)
        self.scale = scale
        self.zero_point = zero_point
        self.bits = bits

    def dequantize(self):
        return [self.zero_point + code * self.scale for code in self.codes]

    def max_weight_error(self, original):
        if len(original) != len(self.codes):
            raise ValueError(
                f"expected {len(self.codes)} weights, got {len(original)}"
            )
        restored = self.dequantize()
        return max(abs(a - b) for a, b in zip(original, restored))

    def quantized_size_bytes(self):
        return math.ceil(len(self.codes) * self.bits / 8)

    def original_size_bytes(self):
        return len(self.codes) * BYTES_PER_FLOAT

    def summary(self, original):
        return {
            "bits": self.bits,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "codes": self.codes,
            "quantized_size_bytes": self.quantized_size_bytes(),
            "original_size_bytes": self.original_size_bytes(),
            "max_weight_error": self.max_weight_error(original),
        }


def quantize_weights(weights, bits=8):
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"unsupported quantization bits: {bits}")
    weights = list(weights)
    if not weights:
        raise ValueError("cannot quantize an empty weight vector")
    low = min(weights)
    high = max(weights)
    levels = 2**bits - 1
    if high == low:
        # Every weight is identical, so the grid collapses to one point
        # and every code is 0; dequantizing restores the value exactly.
        return QuantizedWeights([0] * len(weights), 0.0, low, bits)
    scale = (high - low) / levels
    codes = [round((weight - low) / scale) for weight in weights]
    return QuantizedWeights(codes, scale, low, bits)


class QuantizedModel:
    """A model whose predictions use its dequantized weights."""

    def __init__(self, model, bits=8):
        self.name = model.name
        self.version = model.version
        self.original = model
        self.quantized = quantize_weights(model.weights, bits)
        self.weights = self.quantized.dequantize()
        self.bias = model.bias

    def predict(self, features):
        if len(features) != len(self.weights):
            raise ValueError(
                f"expected {len(self.weights)} features, got {len(features)}"
            )
        return sum(w * x for w, x in zip(self.weights, features)) + self.bias

    def summary(self):
        info = self.original.info()
        info.update(self.quantized.summary(self.original.weights))
        info["quantized"] = True
        return info
