from dataclasses import dataclass
from typing import List

TWEET_ANALYSIS_QUEUE_NAME = "TWEET_ANALYSIS_QUEUE"


@dataclass
class TweetRecord:
    tweet_data: str
    airline_name: str
    sentiment: str


@dataclass
class ClassificationResult:
    airline_name: str
    sentiment: str


@dataclass
class EvaluationResult:
    tweet: TweetRecord
    prediction: ClassificationResult


@dataclass
class AccuracyResult:
    sentiment_accuracy: float
    airline_accuracy: float


@dataclass
class DatasetMetadata:
    tweets: List[TweetRecord]
    unique_airlines: List[str]
