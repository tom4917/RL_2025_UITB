import numpy as np, mujoco
from .Pointing import Pointing     # bestehende Klasse erben

class PointingFixed8(Pointing):
    _sequence = [
        np.array([0.0,  0.25,  0.20]),  # 1
        np.array([0.0, -0.25,  0.20]),  # 2  (Dist ~0.50)
        np.array([0.0,  0.25, -0.20]),  # 3  (Dist ~0.64)
        np.array([0.0, -0.25, -0.20]),  # 4  (Dist ~0.50)
        np.array([0.0,  0.00,  0.00]),  # 5  (Dist ~0.32)
        np.array([0.0,  0.10,  0.30]),  # 6  (Dist ~0.316)
        np.array([0.0, -0.10, -0.30]),  # 7 (Dist ≈ 0.632)
        np.array([0.0,  0.30,  0.00]),  # 8 (Dist ≈ 0.500)
    ]

    def _reset(self, model, data):
        super()._reset(model, data)
        self._max_trials = len(self._sequence)
        return self._info

    def _spawn_target(self, model, data):
        self._target_position = self._sequence[self._trial_idx % len(self._sequence)]
        model.body("target").pos[:] = self._target_origin + self._target_position
        self._target_radius = 0.08
        model.geom("target").size[0] = self._target_radius
        mujoco.mj_forward(model, data)
