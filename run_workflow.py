import asyncio
from temporalio.client import Client
from workflow import EvaluateSentimentWorkflow

from shared import TWEET_ANALYSIS_QUEUE_NAME


async def start():
    client = await Client.connect("localhost:7233")

    handle = await client.start_workflow(
        EvaluateSentimentWorkflow.run,
        "airline_train.csv",  # Customize this
        id="evaluate-sentiment-workflow-003",
        task_queue=TWEET_ANALYSIS_QUEUE_NAME
    )

    accuracy, avg_sentiment = await handle.result()
    print(f"✅ Sentiment Accuracy: {accuracy.sentiment_accuracy:.2%}")
    print(f"✈️ Airline Name Accuracy: {accuracy.airline_accuracy:.2%}")
    print(f"📊 Average Predicted Sentiment Score: {avg_sentiment:.2f}")


asyncio.run(start())
