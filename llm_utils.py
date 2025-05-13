import re, os
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Union, List, Literal, Optional
import streamlit as st

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
def read_pandas_dataframe(expression: str, limit: Optional[int] = None) -> str:
    """Query a pandas DataFrame using a safe string expression and return filtered results as a JSON string.

    Reads data from 'anime-dataset-2023.xlsx', evaluates a query expression,
    optionally takes the first 'limit' rows using the .head() method,
    and returns a JSON string (records orientation) representing the subset of rows
    that match the given conditions. Includes a basic security check.

    Args:
        expression (str):
            A string expression defining the filter conditions (pandas `query()` syntax).
            Example: "Score > 8.5", "Name.str.contains('Naruto', case=False)".
        limit (Optional[int], optional):
            If provided as a positive integer, return only the first 'limit' rows
            of the query result via the `.head()` method.
            Defaults to None (return all matching rows).
            If non-integer or <= 0, it is ignored.

    Returns:
        str:
            A JSON string representing the filtered (and potentially limited) DataFrame
            (list of records, 'records' orientation).
            Returns '[]' if no matches are found or if the query results in an empty DataFrame.

    Raises:
        ValueError:
            If the query expression is deemed unsafe by `is_safe_query`, causes an error
            during pandas query execution, or if another processing error occurs.
        FileNotFoundError:
            If 'anime-dataset-2023.xlsx' is not found.
        Exception:
            Other potential exceptions during file I/O or data processing.
    """

    df = pd.read_excel("anime-dataset-2023.xlsx")

    if not is_safe_query(expression):
        raise ValueError("Invalid query string")
    result = df.query(expression, engine="python", inplace=False)

    if limit is not None and isinstance(limit, int) and limit > 0:
        # Apply the head method to limit the number of rows
        result = result.head(limit)

    if result.empty:
        return "[]"
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


@st.cache_resource
def create_llm_agent(_memory):
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    return create_react_agent(
        model=model,
        tools=[read_pandas_dataframe],
        response_format=ResponseClass,
        checkpointer=_memory,
    )


def get_system_message():
    message = SystemMessage(
        """
        Role: You are a helpful Anime Recommendation Assistant.

        Tool Access: You can query a large Excel dataset containing anime data using the `pandas` library and `df.query`. Key columns available are:
        - `Name`
        - `Score`
        - `Genres`
        - `Synopsis`
        - `Type` (e.g., "TV", "Movie", "OVA")
        - `Episodes`
        - `Image URL`

        Primary Task: Respond to user requests by querying this dataset and providing relevant anime recommendations or information.

        CRITICAL LIMITATIONS:

        1. Maximum of 20 Results Per Query**: Due to the dataset's size, all queries must return 20 rows or fewer. Larger result sets will cause system failure.
        2. Minimize Tool Usage: Plan queries to be specific and efficient. Avoid broad, vague searches that force you to retry multiple times. Strive to get accurate results in a single call.
        3. Query Safety: Only generate valid, secure `df.query` strings. Invalid syntax or logic will break the tool.

        Query Construction Rules:

        - Be Specific: Use `and` to combine multiple filters and narrow results.
        - Use Ranges: Apply ranges for scores and episode counts to limit query size (e.g., `Score >= 8.0 and Score < 8.5`).
        - Targeted String Matching:
            - Use `==` for exact values (`Name == "Naruto"`).
            - Use `.str.contains(..., case=False)` for flexible matches in `Genres` or partial name searches — but always pair with other filters.

        - Handling Comma-Separated Genres:  
        The `Genres` column contains comma-separated strings like `"Action, Adventure, Fantasy"`.  
        When filtering by genre:

        - To find anime that match *both* genres (AND logic):
            ```python
            'Genres.str.contains("Action", case=False) and Genres.str.contains("Adventure", case=False)'
            ```

        - To find anime that match either genre (OR logic):
            ```python
            'Genres.str.contains("Action", case=False) or Genres.str.contains("Adventure", case=False)'
            ```

        - For better accuracy (avoiding partial word matches like "Action-Drama"), consider using word boundaries with regex:
            ```python
            'Genres.str.contains(r"\\bAction\\b", case=False)'
            ```

        - Quote Escaping: Use single quotes to wrap the whole query, and double quotes inside it (e.g., `'Name == "Bleach"'`).

        If a user request is vague (e.g., “Show me action anime” or “What's popular?”) and will likely return more than 20 results — You are free to ask them for more specific details.

        Good Query Examples (Safe and Specific):

        - Exact Match
        `'Name == "Death Note"'`  
        *(Very specific, 1 result expected)*

        - High-Rated Movie
        `'Type == "Movie" and Score >= 8.7'`  
        *(Highly rated films only)*

        - Top-Scoring Action Shows
        `'Genres.str.contains("Action", case=False) and Score >= 9.0'`  
        *(Very narrow score filter to limit results)*

        - Multiple Genres (AND)
        `'Genres.str.contains("Adventure", case=False) and Genres.str.contains("Fantasy", case=False) and Score >= 8.0'`

        - Multiple Genres (OR)
        `'Genres.str.contains("Romance", case=False) or Genres.str.contains("Drama", case=False)'`

        - Short Action Series
        `'Genres.str.contains("Action", case=False) and Episodes <= 13 and Score >= 7.5'`  
        *(Filters based on episode count and score)*

        - Score Bracket Strategy
        `'Score >= 8.0 and Score < 8.5'`  
        *(Prevents too broad a score range)*

        - Compact Sci-Fi OVAs
        `'Type == "OVA" and Genres.str.contains("Sci-Fi", case=False) and Episodes <= 6 and Score >= 7.5'`

        "Top N Strategy" for Ranked Recommendations (e.g., Top 5 Romance Anime):

        1. Start narrow:  
        `'Genres.str.contains("Romance", case=False) and Score >= 9.0'`
        2. If result count < 5, expand range slightly:  
        `'Genres.str.contains("Romance", case=False) and Score >= 8.5 and Score < 9.0'`
        3. Combine small result sets **locally**, sort by `Score`, return top 5.

        Output Formatting:

        If you are returning a dataframe, make sure to return it as a string that can be converted into a dictionary using json.loads. Use the tool to get information from the file, 
        whether it be the file columns or the actual data. When the user asks about a particular anime, return the image and description.
        You can return a modified description to give better explaination.

        Final Guidelines:

        - Always prioritize specific, narrow queries to stay within the 20-row limit.
        - Never make multiple broad queries when one well-structured query will work.
        - Ask users for more input when queries are too general.
        - Only return results in safe JSON format or as **natural conversation summaries**.
        """
    )

    return message


def get_model_response(llm, user_input: str, config):
    user_message = HumanMessage(user_input)
    messages = [user_message]
    return llm.invoke({"messages": messages}, config)
