import csv
import tempfile
import unittest
from datetime import date, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.monitor_raspberry import DailyLog, DailyStats, FIELDS, Metric, Sampler, temperature, throttled


def sample(moment="2026-11-26T12:00:00-05:00", cpu=25, temp=50):
    row = dict.fromkeys(FIELDS, "")
    row.update(momento=moment, intervalo_s=10, cpu_pct=cpu, temperatura_c=temp,
               ram_pct=30, ram_usada_mb=2400, ram_disponible_mb=5600,
               ram_total_mb=8000, nucleos=4, throttled_hex="0x0")
    return row


class PerformanceLogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def test_daily_csv_and_peak_time_and_restart_preserve_history(self):
        output = DailyLog(self.folder)
        output.append(sample(cpu=80, temp=71))
        output.append(sample("2026-11-26T12:00:10-05:00", cpu=20, temp=52))
        output.flush()
        restarted = DailyLog(self.folder)
        restarted.append(sample("2026-11-26T12:00:20-05:00", cpu=5, temp=48))
        text = (self.folder / "2026-11-26.log").read_text(encoding="utf-8")
        self.assertIn("Muestras: 3", text)
        self.assertIn("CPU total (%) | 5.00 | 35.00 | 5.00", text)
        self.assertIn("80.00 | 2026-11-26T12:00:00-05:00", text)
        with (self.folder / "2026-11-26.csv").open(encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 3)

    def test_midnight_rotates_file_and_resets_daily_maximum(self):
        output = DailyLog(self.folder)
        output.append(sample("2026-11-26T23:59:55-05:00", cpu=90))
        output.append(sample("2026-11-27T00:00:05-05:00", cpu=10))
        before = (self.folder / "2026-11-26.log").read_text(encoding="utf-8")
        after = (self.folder / "2026-11-27.log").read_text(encoding="utf-8")
        self.assertIn("90.00", before)
        self.assertNotIn("90.00", after)
        self.assertIn("Muestras: 1", after)

    def test_missing_sensor_is_not_reported_as_zero(self):
        output = DailyLog(self.folder)
        row = sample(temp=None)
        row["throttled_hex"] = None
        output.append(row)
        text = (self.folder / "2026-11-26.log").read_text(encoding="utf-8")
        self.assertIn("Temperatura SoC/CPU (°C) | n/d", text)
        self.assertIn("Bits históricos observados: n/d", text)
        self.assertIn("bajo voltaje: n/d", text)

    def test_current_flags_and_history_are_different(self):
        stats = DailyStats(date(2026, 11, 26))
        row = sample()
        row["throttled_hex"] = "0x50000"  # ocurrió desde el arranque, no está activo
        stats.add(row)
        self.assertEqual(stats.flag_samples[0], 0)
        self.assertEqual(stats.flag_samples[2], 0)
        row["throttled_hex"] = "0x50005"
        stats.add(row)
        self.assertEqual(stats.flag_samples[0], 1)
        self.assertEqual(stats.flag_samples[2], 1)
        self.assertEqual(stats.history, 0x50000)

    def test_invalid_metric_does_not_pollute_mean_or_peak(self):
        metric = Metric()
        for value in (None, "", "nan", "inf"):
            metric.add(value, "bad")
        metric.add(20, "good")
        self.assertEqual((metric.count, metric.total, metric.maximum_at), (1, 20, "good"))

    def test_partial_last_csv_row_is_skipped_and_next_sample_has_its_own_line(self):
        output = DailyLog(self.folder)
        output.append(sample())
        path = self.folder / "2026-11-26.csv"
        with path.open("a", encoding="utf-8") as handle:
            handle.write("2026-11-26T12:00:05-05:00,10,99")
        restarted = DailyLog(self.folder)
        restarted.append(sample("2026-11-26T12:00:10-05:00", cpu=15))
        self.assertEqual(restarted.stats.count, 2)
        self.assertEqual(restarted.stats.metrics["cpu_pct"].maximum, 25)

    def test_incompatible_csv_is_not_overwritten(self):
        path = self.folder / "2026-11-26.csv"
        path.write_text("unrelated,header\n1,2\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            DailyLog(self.folder).append(sample())
        self.assertEqual(path.read_text(encoding="utf-8"), "unrelated,header\n1,2\n")

    def test_summary_waits_five_minutes_but_all_samples_are_saved(self):
        with patch("tools.monitor_raspberry.time.monotonic", return_value=100) as clock:
            output = DailyLog(self.folder, summary_seconds=300)
            output.append(sample())
            path = self.folder / "2026-11-26.log"
            self.assertIn("Muestras: 1", path.read_text(encoding="utf-8"))
            clock.return_value = 110
            output.append(sample("2026-11-26T12:00:10-05:00"))
            self.assertIn("Muestras: 1", path.read_text(encoding="utf-8"))
            clock.return_value = 400
            output.append(sample("2026-11-26T12:05:00-05:00"))
            self.assertIn("Muestras: 3", path.read_text(encoding="utf-8"))


class SamplerTests(unittest.TestCase):
    def test_collector_uses_available_ram_and_normalizes_process_cpu(self):
        memory = SimpleNamespace(total=8000 * 2**20, available=5000 * 2**20, percent=37.5)
        process = SimpleNamespace(pid=123, is_running=lambda: True,
                                 cpu_percent=lambda **_: 200,
                                 memory_info=lambda: SimpleNamespace(rss=400 * 2**20))
        with patch("tools.monitor_raspberry.command", return_value="123"), \
                patch("tools.monitor_raspberry.psutil.cpu_count", return_value=4), \
                patch("tools.monitor_raspberry.psutil.cpu_percent", side_effect=[0, [0]*4, 25, [10,20,30,40]]), \
                patch("tools.monitor_raspberry.psutil.Process", return_value=process), \
                patch("tools.monitor_raspberry.psutil.virtual_memory", return_value=memory), \
                patch("tools.monitor_raspberry.psutil.swap_memory", return_value=SimpleNamespace(percent=0,used=0)), \
                patch("tools.monitor_raspberry.psutil.disk_usage", return_value=SimpleNamespace(percent=40,free=10*2**30)), \
                patch("tools.monitor_raspberry.psutil.cpu_freq", return_value=SimpleNamespace(current=2400)), \
                patch("tools.monitor_raspberry.os.getloadavg", return_value=(1,2,3), create=True), \
                patch("tools.monitor_raspberry.temperature", return_value=65), \
                patch("tools.monitor_raspberry.throttled", return_value=0):
            sampler = Sampler(timezone.utc, Path.cwd())
            row = sampler.collect()
        self.assertEqual(set(row), set(FIELDS))
        self.assertEqual(row["ram_usada_mb"], 3000)
        self.assertEqual(row["camdetector_cpu_pct"], 50)
        self.assertEqual(row["camdetector_ram_mb"], 400)
        self.assertEqual(row["nucleo_max_pct"], 40)

    def test_unavailable_service_is_not_a_zero_cpu_reading(self):
        with patch("tools.monitor_raspberry.command", return_value="0"):
            sampler = Sampler(timezone.utc, Path.cwd())
            self.assertEqual(sampler.process_sample(), (None, None, None))

    def test_raspberry_sensors_fail_gracefully_and_decode_hex(self):
        with patch("tools.monitor_raspberry.Path.read_text", side_effect=FileNotFoundError):
            self.assertIsNone(temperature())
        with patch("tools.monitor_raspberry.command", return_value=None):
            self.assertIsNone(throttled())
        with patch("tools.monitor_raspberry.command", return_value="throttled=0x50005"):
            self.assertEqual(throttled(), 0x50005)


if __name__ == "__main__":
    unittest.main()
