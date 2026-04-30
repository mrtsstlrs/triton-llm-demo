import time

import numpy as np
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
            input_tensor = pb_utils.get_input_tensor_by_name(req, "INPUT")
            if input_tensor is None:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError("Missing required input tensor 'INPUT'")
                    )
                )
                continue

            values = input_tensor.as_numpy()
            raw_value = values.flat[0]
            if isinstance(raw_value, bytes):
                raw_value = raw_value.decode("utf-8")
            output_values = np.array(
                [f"echo: {raw_value}"], dtype=object
            )
            t1 = time.time_ns()
            self.lat.increment(t1 - t0)
            output_tensor = pb_utils.Tensor("OUTPUT", output_values)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        return responses
