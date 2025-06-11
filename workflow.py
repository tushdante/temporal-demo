from typing import List, Tuple
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy
with workflow.unsafe.imports_passed_through():
    from activities import (
        read_csv_activity,
        classify_tweet_activity,
        compute_accuracy_activity,
        compute_average_sentiment_activity
    )
from shared import TweetRecord, EvaluationResult, AccuracyResult, DatasetMetadata
from exceptions import OpenAIError, InvalidJSONResponse, MissingPrediction


@workflow.defn
class EvaluateSentimentWorkflow:
    @workflow.run
    async def run(self, file_path: str) -> Tuple[AccuracyResult, float]:
        metadata: DatasetMetadata = await workflow.execute_activity(
            read_csv_activity,
            file_path,
            start_to_close_timeout=timedelta(seconds=30)
        )

        results: List[EvaluationResult] = []

        for tweet in metadata.tweets:
            classification = await workflow.execute_activity(
                classify_tweet_activity,
                # Tuple: (TweetRecord, List[str])
                (tweet, metadata.unique_airlines),
                start_to_close_timeout=timedelta(seconds=20),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    non_retryable_error_types=[
                        "InvalidJSONResponse", "MissingPrediction"]
                )
            )
            results.append(EvaluationResult(
                tweet=tweet, prediction=classification))

        accuracy: AccuracyResult = await workflow.execute_activity(
            compute_accuracy_activity,
            results,
            start_to_close_timeout=timedelta(seconds=10),
        )

        avg: float = await workflow.execute_activity(
            compute_average_sentiment_activity,
            results,
            start_to_close_timeout=timedelta(seconds=10),
        )

        return (accuracy, avg)
