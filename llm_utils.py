import re, os
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from pandas import DataFrame
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Union, List, Literal, Dict, Any

from pydantic import BaseModel, Field


import environ

env = environ.Env()
environ.Env.read_env()
APIKEY = env("OPENAI_API_KEY")


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = APIKEY


def is_safe_query(query_string):
    """
    Checks query safety before execution

    Return boolean. True if string is safe and false otherwise
    """

    dangerous_keywords = [
        "import",
        "eval",
        "exec",
        "execfile",
        "globals",
        "locals",
        "open",
        "compile",
        "input",
    ]

    if "__" in query_string or ";" in query_string:
        return False

    # Check for potentially dangerous words
    for keyword in dangerous_keywords:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, query_string, re.IGNORECASE):
            return False

    return True


@tool
def read_pandas_dataframe(expression: str) -> str:
    """Query a pandas DataFrame using a safe string expression and return filtered results as a JSON string.

    Reads data from 'anime-sample-dataset.xlsx', evaluates a query expression,
    and returns a JSON string (records orientation) representing the subset of rows
    that match the given conditions. Includes a security check.

    Args:
        expression (str):
            A string expression defining the filter conditions (pandas `query()` syntax).
            Example: "Rating > 8.5", "Name.str.contains('Naruto')".

    Returns:
        str:
            A JSON string representing the filtered DataFrame (list of records).
            Returns '[]' if no matches are found or if the query results in an empty DataFrame.

    Raises:
        ValueError:
            If the query expression is deemed unsafe or causes an error during execution.
            FileNotFoundError: If 'anime-sample-dataset.xlsx' is not found.
    """

    df = pd.read_excel("anime-sample-dataset.xlsx")
    if not is_safe_query(expression):
        raise ValueError("Invalid query string")
    result = df.query(expression, engine="python", inplace=False)

    if result.empty:
        return "[]"  # Explicitly return empty JSON array
    elif isinstance(result, pd.Series):
        return result.to_frame().to_json(orient="records")
    else:
        return result.to_json(orient="records")


class ImageCard(BaseModel):
    """Represents an image card with metadata."""

    title: str = Field(description="Title of the image card")
    description: str = Field(description="Description of the image")
    image_url: str = Field(description="URL of the image")


class ResponseClass(BaseModel):
    """Response to a prompt with dynamic data types."""

    response_type: Literal["text", "image", "dataframe"] = Field(
        description="Type of response (text, image, or dataframe)"
    )

    # Union type for response_data with discriminator based on response_type
    response_data: Union[
        str,  # For "text" type
        List[ImageCard],  # For "image" type (list of image cards)
    ] = Field(..., description="Response data matching the response_type")


def create_llm_agent():
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    return create_react_agent(
        model, [read_pandas_dataframe], response_format=ResponseClass
    )


def get_system_message():
    message = SystemMessage(
        """
        You are an anime recommendation system. You have access to a tool that allows you to get information from an excel file about anime.
        It has information such as names and ratings. Your job is to return the best response based on the information supplied by the user.
        Make sure to come up with correct and safe query expressions to query the dataframe. If you are returning a dataframe,
        make sure to return it as a string that can be converted into a dictionary using json.loads. Use the tool to get information from the file, 
        Whether it be the file columns or the actual data. When the user asks about a particular anime, return the image and description.
        You can return a modified description to give better explaination.
        """
    )

    return message


def get_model_response(user_input: str):
    sytem_message = get_system_message()
    user_message = HumanMessage(user_input)

    messages = [sytem_message, user_message]

    llm = create_llm_agent()

    return llm.invoke({"messages": messages})
