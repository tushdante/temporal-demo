import os
import json
from typing import List, Tuple, Dict
import re
from collections import defaultdict

from shared import (
    TweetRecord,
    ClassificationResult,
    EvaluationResult,
    AccuracyResult,
    DatasetMetadata,
)
from exceptions import OpenAIError, InvalidJSONResponse, MissingPrediction
from temporalio import activity
import pandas as pd
import openai

# Set your OpenAI API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@activity.defn
async def read_csv_activity(file_path: str) -> DatasetMetadata:
    """Parse the input csv file and generate a DatasetMetadata object for use in later activities

    Args:
        file_path (str): Input file path for the csv

    Returns:
        DatasetMetadata: Dataclass containing a list of Tweets and a list of unique airline names
    """
    df = pd.read_csv(
        file_path, header=0, usecols=["text", "airline", "airline_sentiment"]
    )  # read first 50 rows

    df = df.sample(n=100, random_state=42)  # sample a random selection of 100 rows
    
    tweets = [
        TweetRecord(
            tweet_data=row["text"],
            airline_name=row["airline"],
            sentiment=row["airline_sentiment"],
        )
        for _, row in df.iterrows()
    ]

    unique_airlines = list({tweet.airline_name.strip() for tweet in tweets})

    return DatasetMetadata(tweets=tweets, unique_airlines=unique_airlines)


@activity.defn
async def classify_tweet_activity(
    data: Tuple[TweetRecord, List[str]],
) -> ClassificationResult:
    """Use the OpenAI API to extract the airline name and sentiment for a given Tweet.
    We use the list of unique airline names to ensure the model coerces its output to a normalized value for comparison later

    Args:
        data (Tuple[TweetRecord, List[str]]): Tuple of a TweetRecord object and a list of unique airline name

    Raises:
        MissingPrediction: Validate airline name or sentiment is present in the resposne
        OpenAIError: Raised when errors are returned by the API
        InvalidJSONResponse: Validate that the response is a valid JSON object

    Returns:
        ClassificationResult: Result object that contains the predicted airline_name and sentiment
    """
    tweet, airlines = data
    airline_list_str = ", ".join(sorted(airlines))
    prompt = (
        f"Analyze the following tweet and extract the airline name and sentiment.\n"
        f"Tweet: {tweet.tweet_data}\n"
        f"Airline names must be one of: {airline_list_str} (case-insensitive).\n"
        f"Sentiment must be one of: 'Positive', 'Negative', 'Neutral'.\n"
        f"Return the result in JSON format like: {{'airline': 'airline_name', 'sentiment': 'sentiment_value'}}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in sentiment analysis and entity extraction.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        json_str = re.search(r"\{.*\}", content, re.DOTALL).group()
        parsed_data = json.loads(json_str)

        airline = parsed_data.get("airline", "").strip()
        sentiment = parsed_data.get("sentiment", "").strip()

        if not airline or not sentiment:
            raise MissingPrediction("Missing airline or sentiment in response")

        return ClassificationResult(airline_name=airline, sentiment=sentiment)

    except openai.APIConnectionError as e:
        raise OpenAIError(f"OpenAI call failed: {str(e.__cause__)}") from e
    except json.JSONDecodeError as e:
        raise InvalidJSONResponse("Failed to parse JSON from OpenAI response") from e
    except Exception as e:
        raise OpenAIError(f"Unexpected classification error: {str(e)}") from e


@activity.defn
async def compute_average_sentiment_activity(
    results: List[EvaluationResult],
) -> Dict[str, float]:
    """Generate the average sentiment for a given airline. Positive sentiments are scored with +1, negative as -1
    and netural sentiments are scored with 0

    Args:
        results (List[EvaluationResult]): List of objects that contain the TweetRecord and predicted sentiment

    Returns:
        Dict[str, float]: Average value of the sentiment for each airline
    """
    # Initialize dictionaries to track total scores and counts
    score_totals = defaultdict(int)
    score_counts = defaultdict(int)

    # Scoring map
    sentiment_score = {
        "positive": 1,
        "neutral": 0,
        "negative": -1,
    }

    # Aggregate scores by airline
    for result in results:
        airline = result.tweet.airline_name
        sentiment = result.tweet.sentiment.lower()
        score = sentiment_score.get(sentiment, 0)
        score_totals[airline] += score
        score_counts[airline] += 1

    # Compute averages
    average_scores = {
        airline: score_totals[airline] / score_counts[airline]
        for airline in score_totals
        if score_counts[airline] > 0
    }

    return average_scores


@activity.defn
async def compute_accuracy_activity(results: List[EvaluationResult]) -> AccuracyResult:
    """Compute the accuracy of the prediction by comparing against the labeled dataset

    Args:
        results (List[EvaluationResult]): List of objects that contain the TweetRecord and predicted sentiment

    Returns:
        AccuracyResult: Object that contains the accuracy values for sentiment and airline_name
    """
    correct_sentiment = 0
    correct_airline = 0
    total = len(results)

    for r in results:
        if r.tweet.sentiment.lower() == r.prediction.sentiment.lower():
            correct_sentiment += 1
        if r.tweet.airline_name.lower() == r.prediction.airline_name.lower():
            correct_airline += 1

    sentiment_accuracy = correct_sentiment / total if total else 0.0
    airline_accuracy = correct_airline / total if total else 0.0

    return AccuracyResult(sentiment_accuracy, airline_accuracy)

@activity.defn
async def print_sentiment_scores_table_activity(scores: Dict[str, float]) -> None:
    """
    Temporal activity that prints a formatted table of airline sentiment scores.

    Args:
        scores (dict[str, float]): A dictionary mapping airline names to sentiment scores.
    """
    if not scores:
        print("No sentiment scores to display.")
        return

    print("\nSentiment Scores by Airline:\n")
    print(f"{'Airline':<25} | {'Sentiment Score':>16}")
    print("-" * 45)

    for airline, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{airline:<25} | {score:>16.2f}")

    print()
    return
