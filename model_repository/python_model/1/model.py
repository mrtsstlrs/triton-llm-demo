import time
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.lat_family = pb_utils.MetricFamily(
            name="my_preprocess_latency_ns",
            description="Суммарное время на препроцессинг",
            kind=pb_utils.MetricFamily.COUNTER
        )
        self.lat = self.lat_family.Metric(labels={"model": args["model_name"], "version": args["model_version"]})

    def execute(self, requests):
        responses = []
        for req in requests:
            t0 = time.time_ns()
            # инференс...
            t1 = time.time_ns()
            self.lat.increment(t1 - t0)
            # формируем response...
        return responses
