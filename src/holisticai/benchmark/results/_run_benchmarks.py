import os

from holisticai.benchmark import BiasMitigationBenchmark


def main():
    for task_type in ["binary_classification", "regression"]:
       for stage in ["preprocessing", "inprocessing", "postprocessing"]:
            benchmark = BiasMitigationBenchmark(task_type, stage)
            datasets = benchmark.get_datasets()
            mitigators = benchmark.get_mitigators()
            results = benchmark._build_benchmark(datasets=datasets, mitigators=mitigators)
            osdir = os.path.dirname(os.path.abspath(__file__))
            if not os.path.exists(os.path.join(osdir, "bias", task_type, stage)):
                os.makedirs(os.path.join(osdir, "bias", task_type, stage))
            results.to_csv(os.path.join(osdir, "bias", task_type, stage, "benchmark.csv"), index=False)
            print(f"Results saved for {task_type} task type and {stage} stage.")

if __name__ == "__main__":
    main()
