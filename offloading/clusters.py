"""Cluster model for the offloading simulation (Objective 3).

We have no live multi-cluster infrastructure, so the "clusters" are simple
analytic models. The *primary* cluster is the machine the energy data was
measured on (so its energy and runtime are the ground-truth we observed). A
*secondary* cluster is an alternative environment (e.g. greener off-prem / cloud)
described relative to the primary:

* ``energy_factor`` — energy multiplier. < 1 means more energy-efficient hardware,
  so the same job consumes less energy there.
* ``speed_factor``  — speed multiplier. > 1 means faster, so the job's runtime is
  divided by this factor.
* ``cost_per_kwh``  — monetary price per kWh on that cluster. Secondary capacity is
  usually rented, hence typically pricier — this is the tension against the
  energy-efficiency gain.

The defaults model a realistic trade-off: the secondary is greener and a touch
faster, but more expensive per kWh — so blindly offloading everything is not free.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cluster:
    name: str
    energy_factor: float  # energy multiplier relative to measured (primary = 1.0)
    speed_factor: float   # speed multiplier relative to measured (primary = 1.0)
    cost_per_kwh: float   # monetary cost per kWh

    def energy_of(self, primary_energy_wh: float) -> float:
        """Energy (Wh) this cluster would use for a job measured at ``primary_energy_wh``."""
        return primary_energy_wh * self.energy_factor

    def runtime_of(self, primary_runtime_s: float) -> float:
        """Runtime (s) this cluster would take for a job measured at ``primary_runtime_s``."""
        return primary_runtime_s / self.speed_factor

    def cost_of(self, primary_energy_wh: float) -> float:
        """Monetary cost of running a job here, given its primary-measured energy."""
        return self.energy_of(primary_energy_wh) / 1000.0 * self.cost_per_kwh


def default_clusters() -> tuple[Cluster, Cluster]:
    """A (primary, secondary) pair with a realistic energy/cost trade-off."""
    primary = Cluster(name="primary", energy_factor=1.0, speed_factor=1.0, cost_per_kwh=0.30)
    # Secondary: 30% greener and 20% faster, but 1.8x the electricity price.
    secondary = Cluster(name="secondary", energy_factor=0.70, speed_factor=1.20, cost_per_kwh=0.54)
    return primary, secondary
