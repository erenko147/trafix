"""
Metric module tests — verifies all 12 compute_* functions against fixtures.
Run: python tests/mandatory/test_metrics_modules.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import unittest

_FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "utils" / "fixtures"


class TestTravelTime(unittest.TestCase):
    def test_compute(self):
        from tests.metrics.travel_time import compute_travel_time
        r = compute_travel_time(_FIXTURES)
        self.assertIn("mean_travel_time_s", r)
        self.assertGreaterEqual(r["mean_travel_time_s"], 0)
        self.assertGreater(r["total_trips"], 0)

    def test_min_le_mean_le_max(self):
        from tests.metrics.travel_time import compute_travel_time
        r = compute_travel_time(_FIXTURES)
        self.assertLessEqual(r["min_s"], r["mean_travel_time_s"])
        self.assertLessEqual(r["mean_travel_time_s"], r["max_s"])


class TestWaitingTime(unittest.TestCase):
    def test_compute(self):
        from tests.metrics.waiting_time import compute_waiting_time
        r = compute_waiting_time(_FIXTURES)
        self.assertIn("mean_waiting_time_s", r)
        self.assertGreaterEqual(r["mean_waiting_time_s"], 0)

    def test_waiting_le_travel(self):
        from tests.metrics.waiting_time import compute_waiting_time
        from tests.metrics.travel_time  import compute_travel_time
        wt = compute_waiting_time(_FIXTURES)["mean_waiting_time_s"]
        tt = compute_travel_time(_FIXTURES)["mean_travel_time_s"]
        self.assertLessEqual(wt, tt,
                            "Mean waiting time must be ≤ mean travel time")


class TestTimeLoss(unittest.TestCase):
    def test_compute(self):
        from tests.metrics.time_loss import compute_time_loss
        r = compute_time_loss(_FIXTURES)
        self.assertIn("mean_time_loss_s", r)
        self.assertGreaterEqual(r["mean_time_loss_s"], 0)


class TestQueueLength(unittest.TestCase):
    def test_compute(self):
        from tests.metrics.queue_length import compute_queue_length
        r = compute_queue_length(_FIXTURES)
        self.assertIn("mean_halting_per_junction", r)
        self.assertGreaterEqual(r["mean_halting_per_junction"], 0)


class TestThroughput(unittest.TestCase):
    def test_compute(self):
        from tests.metrics.throughput import compute_throughput
        r = compute_throughput(_FIXTURES, sim_duration_s=3600)
        self.assertIn("throughput_veh_per_hr", r)
        self.assertGreaterEqual(r["throughput_veh_per_hr"], 0)
        self.assertIn("arrived_vehicles", r)

    def test_arrived_matches_tripinfo(self):
        from tests.metrics.throughput  import compute_throughput
        from tests.metrics.travel_time import compute_travel_time
        tp = compute_throughput(_FIXTURES, sim_duration_s=3600)
        tt = compute_travel_time(_FIXTURES)
        self.assertLessEqual(tp["arrived_vehicles"], tt["total_trips"] + tp.get("vehicles_still_running", 0) + 1)


class TestNetworkSpeed(unittest.TestCase):
    def test_compute(self):
        from tests.metrics.network_speed import compute_network_speed
        r = compute_network_speed(_FIXTURES)
        self.assertIn("mean_speed_ms", r)
        self.assertGreaterEqual(r["mean_speed_ms"], 0)


class TestTeleports(unittest.TestCase):
    def test_compute(self):
        from tests.metrics.teleports import compute_teleports
        r = compute_teleports(_FIXTURES)
        self.assertIn("total_teleports", r)
        self.assertGreaterEqual(r["total_teleports"], 0)
        self.assertIsInstance(r["total_teleports"], (int, float))


class TestEmissions(unittest.TestCase):
    def test_compute(self):
        from tests.metrics.emissions import compute_emissions
        r = compute_emissions(_FIXTURES)
        self.assertIn("mean_CO2_per_vehicle_mg", r)
        self.assertGreaterEqual(r["mean_CO2_per_vehicle_mg"], 0)


class TestFuelConsumption(unittest.TestCase):
    def test_compute_volumetric(self):
        from tests.metrics.fuel_consumption import compute_fuel_consumption
        r = compute_fuel_consumption(_FIXTURES, volumetric=True)
        self.assertIn("mean_fuel_per_vehicle_L", r)
        self.assertGreaterEqual(r["mean_fuel_per_vehicle_L"], 0)

    def test_fuel_positive_when_vehicles_moved(self):
        from tests.metrics.fuel_consumption import compute_fuel_consumption
        from tests.metrics.travel_time      import compute_travel_time
        r  = compute_fuel_consumption(_FIXTURES, volumetric=True)
        tt = compute_travel_time(_FIXTURES)
        if tt["total_trips"] > 0:
            self.assertGreater(r["mean_fuel_per_vehicle_L"], 0,
                              "Vehicles that completed trips must have consumed fuel")


class TestPollutants(unittest.TestCase):
    def test_compute(self):
        from tests.metrics.pollutants import compute_pollutants
        r = compute_pollutants(_FIXTURES)
        self.assertIn("totals_mg", r)
        self.assertIn("NOx", r["totals_mg"])

    def test_all_pollutants_present(self):
        from tests.metrics.pollutants import compute_pollutants
        r = compute_pollutants(_FIXTURES)
        for pollutant in ("NOx", "PMx", "HC", "CO"):
            self.assertIn(pollutant, r["totals_mg"],
                         f"Missing pollutant: {pollutant}")
            self.assertGreaterEqual(r["totals_mg"][pollutant], 0)


class TestStopsPerVehicle(unittest.TestCase):
    def test_compute(self):
        from tests.metrics.stops_per_vehicle import compute_stops_per_vehicle
        r = compute_stops_per_vehicle(_FIXTURES)
        self.assertIn("avg_stops_per_vehicle", r)
        self.assertGreaterEqual(r["avg_stops_per_vehicle"], 0)


class TestJunctionFairness(unittest.TestCase):
    def test_compute(self):
        from tests.metrics.junction_fairness import compute_junction_fairness
        r = compute_junction_fairness(_FIXTURES)
        self.assertIn("network_wide_variance", r)
        self.assertGreaterEqual(r["network_wide_variance"], 0)

    def test_variance_finite(self):
        from tests.metrics.junction_fairness import compute_junction_fairness
        import math
        r = compute_junction_fairness(_FIXTURES)
        self.assertFalse(math.isnan(r["network_wide_variance"]),
                        "Junction fairness variance must be finite")


class TestAllMetricsReturnDicts(unittest.TestCase):
    """Smoke test: every metric module returns a non-empty dict."""

    def _check(self, func, *args):
        result = func(_FIXTURES, *args)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_all_return_dicts(self):
        from tests.metrics.travel_time      import compute_travel_time
        from tests.metrics.waiting_time     import compute_waiting_time
        from tests.metrics.time_loss        import compute_time_loss
        from tests.metrics.queue_length     import compute_queue_length
        from tests.metrics.throughput       import compute_throughput
        from tests.metrics.network_speed    import compute_network_speed
        from tests.metrics.teleports        import compute_teleports
        from tests.metrics.emissions        import compute_emissions
        from tests.metrics.fuel_consumption import compute_fuel_consumption
        from tests.metrics.pollutants       import compute_pollutants
        from tests.metrics.stops_per_vehicle import compute_stops_per_vehicle
        from tests.metrics.junction_fairness import compute_junction_fairness

        self._check(compute_travel_time)
        self._check(compute_waiting_time)
        self._check(compute_time_loss)
        self._check(compute_queue_length)
        self._check(compute_throughput, 3600)
        self._check(compute_network_speed)
        self._check(compute_teleports)
        self._check(compute_emissions)
        self._check(compute_fuel_consumption)
        self._check(compute_pollutants)
        self._check(compute_stops_per_vehicle)
        self._check(compute_junction_fairness)


if __name__ == "__main__":
    unittest.main(verbosity=2)
