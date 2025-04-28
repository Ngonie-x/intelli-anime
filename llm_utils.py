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
def read_pandas_dataframe(expression: str) -> DataFrame:
    """Query a pandas DataFrame using a safe string expression and return filtered results.

    This tool evaluates a query expression on a pandas DataFrame and returns a subset of rows
    that match the given conditions. It includes a security check to prevent unsafe queries.

    Args:
        expression (str):
            A string expression defining the filter conditions. Must follow pandas `query()` syntax.
            Example: "age > 30", "name.str.contains('Alice')".

    Returns:
        pd.DataFrame:
            A new DataFrame containing only the rows that satisfy the query conditions.
            Returns an empty DataFrame if no matches are found.

    Raises:
        ValueError:
            If the query expression is deemed unsafe (e.g., contains malicious code or invalid syntax).
    """

    df = pd.read_excel("anime-sample-dataset.xlsx")
    if not is_safe_query(expression):
        raise ValueError("Invalid query string")
    return df.query(expression, inplace=False)


class ImageCard(BaseModel):
    """Represents an image card with metadata."""

    title: str = Field(description="Title of the image card")
    description: str = Field(description="Description of the image")
    image_url: str = Field(description="URL of the image")


class DataFrameDict(BaseModel):
    """Dictionary representation of a DataFrame."""

    # Define a specific structure expected for dataframe data
    data: List[Dict[str, Any]] = Field(
        ..., description="List of records, where each record is a row in the dataframe"
    )


class ResponseClass(BaseModel):
    """Response to a prompt with dynamic data types."""

    response_type: Literal["text", "image", "dataframe"] = Field(
        description="Type of response (text, image, or dataframe)"
    )

    # Union type for response_data with discriminator based on response_type
    response_data: Union[
        str,  # For "text" type
        List[ImageCard],  # For "image" type (list of image cards)
        DataFrameDict,  # For "dataframe" type - with explicit schema
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
        Make sure to come up with correct and safe query expressions to query the dataframe.
        """
    )

    return message


def get_model_response(user_input: str):
    sytem_message = get_system_message()
    user_message = HumanMessage(user_input)

    messages = [sytem_message, user_message]

    llm = create_llm_agent()

    return llm.invoke({"messages": messages})
