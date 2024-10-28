import os  # noqa: INP001

from holisticai.benchmark import BiasMitigationBenchmark


def main():
    for task_type in ["clustering"]:
        for stage in ["preprocessing", "inprocessing", "postprocessing"]:
            benchmark = BiasMitigationBenchmark(task_type, stage)
            results = benchmark.run()
            osdir = os.path.dirname(os.path.abspath(__file__))
            if not os.path.exists(os.path.join(osdir, "bias", task_type, stage)):
                os.makedirs(os.path.join(osdir, "bias", task_type, stage))
            results.to_csv(os.path.join(osdir, "bias", task_type, stage, "benchmark.csv"), index=True)
            print(f"Results saved for {task_type} task type and {stage} stage.")  # noqa: T201


if __name__ == "__main__":
    main()
