import sys
import time
from collections import OrderedDict
from time import gmtime, strftime


class Logging:
    def __init__(self, log_params, total_iterations=None, epochs=None, logger_format="iteration"):
        self.epochs = epochs
        self.total_iterations = total_iterations
        self.logger_format = logger_format
        self.log_params = log_params
        self.time_start = time.time()

    def update(self, *args, **kargs):  # noqa: ARG002
        init = False
        finish = False
        params = OrderedDict()
        for param_value, (param_name, param_type) in zip(args, self.log_params):
            if param_name == "iteration":
                if self.logger_format == "iteration":
                    params["iter"] = f"{int(param_value)}/{int(self.total_iterations)}"
                    init = int(param_value) == 0
                    finish = int(param_value) == int(self.total_iterations)
                elif self.logger_format == "epochs":
                    epoch = (param_value / self.total_iterations) * self.epochs
                    params["epochs"] = f"{ epoch: .2f}"
                    init = param_value == 0
                    finish = int(param_value) == int(self.total_iterations)

            elif param_type is int:
                params[param_name] = param_value

            elif param_type is float:
                params[param_name] = f"{param_value:.4f}"

            elif param_type is str:
                params[param_name] = f"{param_value}"

        current_time = time.time()
        elapsed_time = current_time - self.time_start
        elapsed_ftime = strftime("%H:%M:%S", gmtime(elapsed_time))

        content = f"elapsed time: {elapsed_ftime}"
        for key, value in params.items():
            content += f" | {key}:{value}"
        content = f"[{content}]\r"

        if init:
            content = f"\n{content}"

        if finish:
            content = f"{content}\n"

        sys.stdout.write(content)
        sys.stdout.flush()

    def info(self, message):
        current_time = time.time()
        elapsed_time = current_time - self.time_start
        elapsed_ftime = strftime("%H:%M:%S", gmtime(elapsed_time))

        content = f"elapsed time: {elapsed_ftime}"
        content += f" | {message}"
        content = f"\n[{content}]\n"

        sys.stdout.write(content)
        sys.stdout.flush()
