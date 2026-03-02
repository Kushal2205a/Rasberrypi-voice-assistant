import time
import psutil, threading, os


class ResourceSampler:

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = float(interval)
        self._running = False
        self._thread = None
        self.samples: list = []  # [(timestamp, rss_bytes, cpu_percent), ...]
        if psutil:
            self._proc = psutil.Process(os.getpid())
        else:
            self._proc = None

    def _loop(self) -> None:
        try:
            if self._proc:
                self._proc.cpu_percent(interval=None)  # prime the baseline; first call always returns 0
        except Exception:
            pass
        while self._running:
            try:
                if self._proc:
                    rss = self._proc.memory_info().rss
                    cpu = self._proc.cpu_percent(interval=None)
                else:
                    rss = 0
                    cpu = 0.0
                self.samples.append((time.time(), rss, cpu))
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def summary(self) -> dict:
        if not self.samples:
            return {"peak_rss": 0, "avg_rss": 0, "peak_cpu": 0.0, "avg_cpu": 0.0, "samples": 0}
        rss_vals = [s[1] for s in self.samples]
        cpu_vals = [s[2] for s in self.samples]
        return {
            "peak_rss": max(rss_vals),
            "avg_rss": sum(rss_vals) / len(rss_vals),
            "peak_cpu": max(cpu_vals),
            "avg_cpu": sum(cpu_vals) / len(cpu_vals),
            "samples": len(self.samples),
        }