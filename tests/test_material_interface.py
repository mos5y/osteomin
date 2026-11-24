import unittest
from dataclasses import replace

from src.config import ABMConfig
from src.material import GelMACondition, GelMAPropertyModel


class MaterialInterfaceTests(unittest.TestCase):
    def test_property_model_returns_framework_output(self):
        condition = GelMACondition(10.0, 50.0, 60.0)
        properties = GelMAPropertyModel().predict(condition)

        self.assertGreater(properties.effective_modulus_kpa, 0.0)
        self.assertGreaterEqual(properties.commitment_hazard_multiplier, 0.0)

    def test_material_multiplier_can_be_applied_to_abm_config(self):
        condition = GelMACondition(10.0, 50.0, 60.0)
        model = GelMAPropertyModel(modulus_to_commitment_slope=1.0)
        properties = model.predict(condition)
        effective_config = replace(
            ABMConfig(commit_k_hourly=0.1),
            commit_k_hourly=0.1 * properties.commitment_hazard_multiplier,
        )

        self.assertAlmostEqual(
            effective_config.commit_k_hourly,
            0.1 * properties.commitment_hazard_multiplier,
        )


if __name__ == "__main__":
    unittest.main()

