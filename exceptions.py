class OpenAIError(Exception):
    """Raised when OpenAI API call fails or model doesn't return usable output."""


class InvalidJSONResponse(OpenAIError):
    """Raised when response from OpenAI is not valid JSON."""


class MissingPrediction(OpenAIError):
    """Raised if prediction fields are empty or nonsensical."""
