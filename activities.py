import ast
import os
import json
from typing import List, Tuple
from temporalio import activity
import pandas as pd
import openai
import re
from shared import TweetRecord, ClassificationResult, EvaluationResult, AccuracyResult, DatasetMetadata
from exceptions import OpenAIError, InvalidJSONResponse, MissingPrediction

# Set your OpenAI API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def parse_airline(value) -> str:
    """
    Parses the airline name from a stringified list.
    Returns the first airline string if present, else returns an empty string.
    """
    try:
        items = ast.literal_eval(value)
        if isinstance(items, list) and items:
            return str(items[0])
        else:
            return ""
    except (ValueError, SyntaxError):
        return ""


@activity.defn
async def read_csv_activity(file_path: str) -> DatasetMetadata:
    df = pd.read_csv(file_path,
                     header=0,
                     nrows=10,
                     names=["tweet", "airline", "sentiment"])  # read first 50 rows

    # Apply the parse_airline function to extract the airline name
    df["airline"] = df["airline"].apply(parse_airline)

    tweets = [
        TweetRecord(
            tweet_data=row["tweet"],
            airline_name=row["airline"],
            sentiment=row["sentiment"]
        )
        for _, row in df.iterrows()
    ]

    unique_airlines = list({tweet.airline_name.strip() for tweet in tweets})

    return DatasetMetadata(tweets=tweets, unique_airlines=unique_airlines)


@activity.defn
async def classify_tweet_activity(data: Tuple[TweetRecord, List[str]]) -> ClassificationResult:
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
                {"role": "system", "content": "You are an expert in sentiment analysis and entity extraction."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        json_str = re.search(r'\{.*\}', content, re.DOTALL).group()
        parsed_data = json.loads(json_str)

        airline = parsed_data.get("airline", "").strip()
        sentiment = parsed_data.get("sentiment", "").strip()

        if not airline or not sentiment:
            raise MissingPrediction("Missing airline or sentiment in response")

        return ClassificationResult(airline_name=airline, sentiment=sentiment)

    except openai.APIConnectionError as e:
        raise OpenAIError(f"OpenAI call failed: {str(e.__cause__)}") from e
    except json.JSONDecodeError as e:
        raise InvalidJSONResponse(
            "Failed to parse JSON from OpenAI response") from e
    except Exception as e:
        raise OpenAIError(f"Unexpected classification error: {str(e)}") from e


@activity.defn
async def compute_average_sentiment_activity(results: List[EvaluationResult]) -> float:
    sentiment_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }
    total = sum(sentiment_map.get(r.prediction.sentiment.lower(), 0)
                for r in results)
    avg = total / len(results) if results else 0.0
    return avg


@activity.defn
async def compute_accuracy_activity(results: List[EvaluationResult]) -> AccuracyResult:
    correct_sentiment = 0
    correct_airline = 0
    total = len(results)

    for r in results:
        print(r.tweet.airline_name.lower())
        print(r.prediction.airline_name.lower())
        if r.tweet.sentiment.lower() == r.prediction.sentiment.lower():
            correct_sentiment += 1
        if r.tweet.airline_name.lower() == r.prediction.airline_name.lower():
            correct_airline += 1

    sentiment_accuracy = correct_sentiment / total if total else 0.0
    airline_accuracy = correct_airline / total if total else 0.0

    return AccuracyResult(sentiment_accuracy, airline_accuracy)
