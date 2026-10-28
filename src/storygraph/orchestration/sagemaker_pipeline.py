"""SageMaker Pipeline definition for model training and registration.

Steps: build ranker training set (Spark) -> train two-tower -> train LightGBM ranker ->
offline evaluation against the collaborative-filtering baseline -> conditional model registration.
The content-understanding batch work is orchestrated by Step Functions (see infra/step_functions);
this pipeline consumes its outputs from the feature store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineSpec:
    role_arn: str
    image_uri: str
    bucket: str
    region: str = "eu-central-1"
    instance_gpu: str = "ml.g5.2xlarge"
    instance_cpu: str = "ml.m5.4xlarge"
    min_ndcg_lift: float = 0.05


def build_pipeline(spec: PipelineSpec) -> Any:  # pragma: no cover - needs the SageMaker SDK + AWS
    try:
        from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
        from sagemaker.workflow.condition_step import ConditionStep
        from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
        from sagemaker.workflow.functions import JsonGet
        from sagemaker.workflow.parameters import ParameterString
        from sagemaker.workflow.pipeline import Pipeline
        from sagemaker.workflow.properties import PropertyFile
        from sagemaker.workflow.steps import ProcessingStep
    except ImportError as exc:
        raise ImportError("install storygraph[aws]") from exc

    cutoff = ParameterString(name="Cutoff", default_value="2026-09-01")
    model_version = ParameterString(name="ModelVersion", default_value="two_tower_lgbm_v3")

    def processor(instance: str) -> ScriptProcessor:
        return ScriptProcessor(
            image_uri=spec.image_uri,
            command=["python3"],
            role=spec.role_arn,
            instance_type=instance,
            instance_count=1,
        )

    features = ProcessingStep(
        name="BuildRankerDataset",
        processor=processor(spec.instance_cpu),
        code="scripts/build_ranker_dataset.py",
        job_arguments=["--cutoff", cutoff],
        outputs=[
            ProcessingOutput(
                output_name="dataset",
                source="/opt/ml/processing/output",
                destination=f"s3://{spec.bucket}/ranker-dataset",
            )
        ],
    )
    two_tower = ProcessingStep(
        name="TrainTwoTower",
        processor=processor(spec.instance_gpu),
        code="scripts/train_recsys.py",
        job_arguments=["--stage", "two_tower", "--version", model_version],
        inputs=[
            ProcessingInput(
                source=features.properties.ProcessingOutputConfig.Outputs["dataset"].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="towers",
                source="/opt/ml/processing/model",
                destination=f"s3://{spec.bucket}/models/two-tower",
            )
        ],
    )
    ranker = ProcessingStep(
        name="TrainRanker",
        processor=processor(spec.instance_cpu),
        code="scripts/train_recsys.py",
        job_arguments=["--stage", "ranker", "--version", model_version],
        inputs=[
            ProcessingInput(
                source=features.properties.ProcessingOutputConfig.Outputs["dataset"].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            ),
            ProcessingInput(
                source=two_tower.properties.ProcessingOutputConfig.Outputs["towers"].S3Output.S3Uri,
                destination="/opt/ml/processing/towers",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="ranker",
                source="/opt/ml/processing/model",
                destination=f"s3://{spec.bucket}/models/ranker",
            )
        ],
    )
    report = PropertyFile(name="EvalReport", output_name="report", path="report.json")
    evaluate = ProcessingStep(
        name="OfflineEvaluation",
        processor=processor(spec.instance_cpu),
        code="scripts/evaluate.py",
        job_arguments=["--baseline", "item_cf", "--cutoff", cutoff],
        inputs=[
            ProcessingInput(
                source=ranker.properties.ProcessingOutputConfig.Outputs["ranker"].S3Output.S3Uri,
                destination="/opt/ml/processing/model",
            )
        ],
        outputs=[ProcessingOutput(output_name="report", source="/opt/ml/processing/report")],
        property_files=[report],
    )
    register = ProcessingStep(
        name="RegisterModel",
        processor=processor(spec.instance_cpu),
        code="scripts/register_model.py",
        job_arguments=["--version", model_version],
    )
    gate = ConditionStep(
        name="NDCGGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluate.name, property_file=report, json_path="lift.ndcg@10"
                ),
                right=spec.min_ndcg_lift,
            )
        ],
        if_steps=[register],
        else_steps=[],
    )
    return Pipeline(
        name="storygraph-recsys-training",
        parameters=[cutoff, model_version],
        steps=[features, two_tower, ranker, evaluate, gate],
    )
