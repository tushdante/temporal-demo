from typing import List, Dict
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities import (
        read_csv_activity,
        classify_tweet_activity,
        compute_accuracy_activity,
        compute_average_sentiment_activity,
        print_sentiment_scores_table_activity,
    )
from shared import EvaluationResult, AccuracyResult, DatasetMetadata


@workflow.defn
class EvaluateSentimentWorkflow:
    @workflow.run
    async def run(self, file_path: str) -> AccuracyResult:
        """Workflow runner to take an input csv and emit the accuracy
           of the OpenAI prediction against the labelled text

        Args:
            file_path (str): Input file path

        Returns:
            Tuple[AccuracyResult, Dict]: Tuple of the Accuracy and average sentiment per airline
        """
        # 1. Parse the input csv
        metadata: DatasetMetadata = await workflow.execute_activity(
            read_csv_activity, file_path, start_to_close_timeout=timedelta(seconds=30)
        )

        results: List[EvaluationResult] = []
        # 2. For each Tweet in the csv use OpenAI to predict the sentiment and airline_name
        for tweet in metadata.tweets:
            classification = await workflow.execute_activity(
                classify_tweet_activity,
                # Tuple: (TweetRecord, List[str])
                (tweet, metadata.unique_airlines),
                start_to_close_timeout=timedelta(seconds=20),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    non_retryable_error_types=[
                        "InvalidJSONResponse",
                        "MissingPrediction",
                    ],
                ),
            )
            results.append(EvaluationResult(tweet=tweet, prediction=classification))

        # 3. Compute the accuracy of the prediction
        accuracy: AccuracyResult = await workflow.execute_activity(
            compute_accuracy_activity,
            results,
            start_to_close_timeout=timedelta(seconds=10),
        )

        # 4. Compute the average sentiment for a given airline
        sentiment_scores: Dict[str, float] = await workflow.execute_activity(
            compute_average_sentiment_activity,
            results,
            start_to_close_timeout=timedelta(seconds=10),
        )

        # 5. Print results
        await workflow.execute_activity(
            print_sentiment_scores_table_activity,
            arg=sentiment_scores,
            start_to_close_timeout=timedelta(seconds=10)
        )

        return accuracy
