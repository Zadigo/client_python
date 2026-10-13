from __future__ import annotations

from collections.abc import Sequence
from functools import total_ordering
from typing import Any, NamedTuple

from pydantic import BaseModel, Field
from pydantic_core import core_schema


@total_ordering
class Timestamp:
    """A nanosecond-resolution timestamp."""

    def __init__(self, sec: float, nsec: float) -> None:
        if nsec < 0 or nsec >= 1e9:
            raise ValueError(f"Invalid value for nanoseconds in Timestamp: {nsec}")
        if sec < 0:
            nsec = -nsec
        self.sec: int = int(sec)
        self.nsec: int = int(nsec)

    def __str__(self) -> str:
        return f"{self.sec}.{self.nsec:09d}"

    def __repr__(self) -> str:
        return f"Timestamp({self.sec}, {self.nsec})"

    def __float__(self) -> float:
        return float(self.sec) + float(self.nsec) / 1e9

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Timestamp) and self.sec == other.sec and self.nsec == other.nsec

    # def __ne__(self, other: object) -> bool:
    #     return not self == other

    def __gt__(self, other: Timestamp) -> bool:
        return self.nsec > other.nsec if self.sec == other.sec else self.sec > other.sec

    # def __lt__(self, other: Timestamp) -> bool:
    #     return self.nsec < other.nsec if self.sec == other.sec else self.sec < other.sec

    @classmethod
    def validate(cls, value: Any) -> Timestamp:
        """Validator to convert allowed types into a Timestamp instance."""
        if isinstance(value, cls):
            return value

        # Handle string inputs (e.g., "1697210000.123456789")
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                raise ValueError("Timestamp string must be a valid float-like representation")

        # Handle float/int inputs
        if isinstance(value, (int, float)):
            sec = int(value)
            nsec = round((value - sec) * 1e9)
            return cls(sec, nsec)

        raise TypeError("Expected Timestamp, float, int, or string")

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> core_schema.CoreSchema:
        validation_schema = core_schema.chain_schema([
            core_schema.any_schema(),
            core_schema.no_info_plain_validator_function(cls.validate),
        ])

        serialization_schema = core_schema.plain_serializer_function_ser_schema(
            lambda instance: str(instance), 
            return_schema=core_schema.str_schema()
        )

        return core_schema.json_or_python_schema(
            python_schema=validation_schema,
            json_schema=validation_schema,
            serialization=serialization_schema,
        )


# BucketSpan is experimental and subject to change at any time.
class BucketSpan(NamedTuple):
    offset: int
    length: int


# Timestamp and exemplar are optional.
# Value can be an int or a float.
# Timestamp can be a float containing a unixtime in seconds,
# a Timestamp object, or None.
# Exemplar can be an Exemplar object, or None.
class Exemplar(NamedTuple):
    labels: dict[str, str]
    value: float
    timestamp: float | Timestamp | None = None


# NativeHistogram is experimental and subject to change at any time.
class NativeHistogram(NamedTuple):
    count_value: float
    sum_value: float
    schema: int
    zero_threshold: float
    zero_count: float
    pos_spans: Sequence[BucketSpan] | None = None
    neg_spans: Sequence[BucketSpan] | None = None
    pos_deltas: Sequence[int] | None = None
    neg_deltas: Sequence[int] | None = None
    nh_exemplars: Sequence[Exemplar] | None = None


class _BucketSpan(BaseModel):
    offset: int = Field(...)
    length: int = Field(...)


class Sample(NamedTuple):
    name: str
    labels: dict[str, str] | None
    value: float | None
    timestamp: float | Timestamp | None = None
    exemplar: Exemplar | None = None
    native_histogram: NativeHistogram | None = None


class _Exemplar(BaseModel):
    labels: dict[str, str] = Field(default_factory=dict)
    value: float = Field(...)
    timestamp: float | Timestamp | None = Field(None)


class _NativeHistogram(BaseModel):
    count_value: float = Field(...)
    sum_value: float = Field(...)
    schema: int = Field(...)
    zero_threshold: float = Field(...)
    zero_count: float = Field(...)
    pos_spans: Sequence[BucketSpan] | None = Field(None)
    neg_spans: Sequence[BucketSpan] | None = Field(None)
    pos_deltas: Sequence[int] | None = Field(None)
    neg_deltas: Sequence[int] | None = Field(None)
    nh_exemplars: Sequence[Exemplar] | None = Field(None)


class _Sample(BaseModel):
    name: str = Field(...)
    labels: dict[str, str] = Field(default_factory=dict)
    value: float | None = Field(None)
    timestamp: float | Timestamp | None = Field(None)
    exemplar: Exemplar | None = Field(None)
    native_histogram: NativeHistogram | None = Field(None)
