from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp
from typing import Any, Mapping, Protocol

__all__ = [
    "GelMACondition",
    "MaterialProperties",
    "MaterialPropertyModel",
    "GelMAPropertyModel",
]


@dataclass(frozen=True, slots=True)
class GelMACondition:
    """Experimental material conditions passed to a property model."""

    gelma_concentration_wv_percent: float
    degree_of_methacrylation_percent: float
    exposure_time_seconds: float
    light_intensity_mw_per_cm2: float = 5.0
    wavelength_nm: float = 365.0
    photoinitiator_concentration_wv_percent: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "gelma_concentration_wv_percent",
            "degree_of_methacrylation_percent",
            "exposure_time_seconds",
            "light_intensity_mw_per_cm2",
            "wavelength_nm",
            "photoinitiator_concentration_wv_percent",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class MaterialProperties:
    """Model output used by the simulator boundary."""

    effective_modulus_kpa: float
    commitment_hazard_multiplier: float = 1.0
    model_name: str = "unknown"

    def __post_init__(self) -> None:
        if self.effective_modulus_kpa < 0.0:
            raise ValueError("effective_modulus_kpa must be non-negative")
        if self.commitment_hazard_multiplier < 0.0:
            raise ValueError("commitment_hazard_multiplier must be non-negative")


class MaterialPropertyModel(Protocol):
    """Interface implemented by replaceable GelMA property predictors."""

    name: str

    def predict(self, condition: GelMACondition) -> MaterialProperties:
        ...


@dataclass(frozen=True, slots=True)
class GelMAPropertyModel:
    """Small, replaceable baseline model for framework integration.

    This is intentionally a transparent baseline, not a validated literature
    fit. Replace it with a fitted predictor without changing OsteoModel.
    """

    reference_modulus_kpa: float = 15.0
    concentration_exponent: float = 2.0
    degree_exponent: float = 1.0
    exposure_saturation_seconds: float = 60.0
    modulus_to_commitment_slope: float = 0.5
    name: str = "gelma-baseline-v1"

    def predict(self, condition: GelMACondition) -> MaterialProperties:
        concentration_factor = max(condition.gelma_concentration_wv_percent / 10.0, 0.0)
        degree_factor = max(condition.degree_of_methacrylation_percent / 50.0, 0.0)
        exposure_factor = 1.0 - exp(
            -condition.exposure_time_seconds / self.exposure_saturation_seconds
        )
        modulus = (
            self.reference_modulus_kpa
            * concentration_factor**self.concentration_exponent
            * degree_factor**self.degree_exponent
            * exposure_factor
        )
        multiplier = 1.0 + self.modulus_to_commitment_slope * (
            modulus / self.reference_modulus_kpa - 1.0
        )
        return MaterialProperties(
            effective_modulus_kpa=max(0.0, modulus),
            commitment_hazard_multiplier=max(0.0, multiplier),
            model_name=self.name,
        )


def material_metadata(
    condition: GelMACondition | None,
    properties: MaterialProperties | None,
    model: MaterialPropertyModel | None,
) -> Mapping[str, Any]:
    return {
        "condition": asdict(condition) if condition is not None else None,
        "properties": asdict(properties) if properties is not None else None,
        "model": getattr(model, "name", None) if model is not None else None,
    }

