import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from workflow import EvaluateSentimentWorkflow
from activities import (
    read_csv_activity,
    classify_tweet_activity,
    compute_accuracy_activity,
    compute_average_sentiment_activity,
)

from shared import TWEET_ANALYSIS_QUEUE_NAME


async def main():
    client = await Client.connect("localhost:7233")

    worker = Worker(
        client,
        task_queue=TWEET_ANALYSIS_QUEUE_NAME,
        workflows=[EvaluateSentimentWorkflow],
        activities=[
            read_csv_activity,
            classify_tweet_activity,
            compute_accuracy_activity,
            compute_average_sentiment_activity,
        ],
    )

    print("🚀 Worker started. Listening for workflows...")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
