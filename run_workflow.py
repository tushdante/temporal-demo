import asyncio
import uuid
from temporalio.client import Client
from workflow import EvaluateSentimentWorkflow

from shared import TWEET_ANALYSIS_QUEUE_NAME


async def start():
    client = await Client.connect("localhost:7233")

    workflow_id = f"tweet-classification-workflow-{uuid.uuid4()}"
    
    handle = await client.start_workflow(
        EvaluateSentimentWorkflow.run,
        "airline_train.csv",  # Customize this
        id=workflow_id,
        task_queue=TWEET_ANALYSIS_QUEUE_NAME,
    )

    accuracy = await handle.result()
    print(f"Sentiment Accuracy: {accuracy.sentiment_accuracy:.2%}")
    print(f"Airline Name Accuracy: {accuracy.airline_accuracy:.2%}")


asyncio.run(start())
