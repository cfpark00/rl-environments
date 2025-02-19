from typing import List, Dict, Type
import warnings
from pydantic import BaseModel, ValidationError


def get_json(text: str, json_model: Type[BaseModel]) -> BaseModel:
    """
    Parse a JSON string into an instance of a Pydantic model.

    This function takes a JSON-formatted string (`text`) and attempts to parse and validate it
    against the schema defined by the provided Pydantic model class (`json_model`). If parsing
    or validation fails, a ValidationError is raised after emitting a warning.

    Args:
        text (str): The JSON string to be parsed.
        json_model (Type[BaseModel]): A Pydantic model class representing the expected data schema.
                                      For example, if you have a model `User(BaseModel)`, you can pass it here.

    Returns:
        BaseModel: An instance of the provided Pydantic model containing the parsed data.

    Raises:
        ValidationError: If the JSON string cannot be parsed or does not conform to the expected schema.
    """
    try:
        # Use Pydantic V2's model_validate_json method to parse and validate the JSON string.
        return json_model.model_validate_json(text)
    except ValidationError as e:
        # Warn about the validation error before re-raising it.
        warnings.warn(f"Validation error during JSON parsing: {e}")
        raise e


def check_messages(messages: List[Dict], system_key: str = "system") -> bool:
    """
    Check if the messages are in the correct format. The message should optionally begin with a system/developer message, followed by an alternating sequence of user and assistant messages.

    Args:
    messages: list of messages in the format [{"role": "user"|"assistant"|"system", "content": <message_content>}]
    system_key: the key used to denote system messages, Default: "system"

    Returns:
    bool: True if the messages are in the correct format, False otherwise
    """
    if len(messages) == 0:
        return False
    n_system_messages = 0
    for i_message, message in enumerate(messages):
        if i_message == 0:
            if message["role"] == system_key:
                n_system_messages = 1
            elif message["role"] != "user":
                return False
        else:
            role = message["role"]
            if role not in ["user", "assistant"]:
                return False
            if (i_message - n_system_messages) % 2 == 0:
                if role != "user":
                    return False
            else:
                if role != "assistant":
                    return False
    return True


def get_n_assistant_messages(messages: List[Dict]) -> int:
    """
    Get the number of assistant messages in the conversation.

    Args:
    messages: list of messages in the format [{"role": "user"|"assistant", "content": <message_content>}]

    Returns:
    int: number of assistant messages in the conversation
    """
    check_messages(messages)
    n_assistant_messages = 0
    for message in messages:
        if message["role"] == "assistant":
            n_assistant_messages += 1
    return n_assistant_messages


def get_last_assistant_message(
    messages: List[Dict], system_key: str = "system"
) -> Dict:
    """
    Get the last assistant message in the conversation.

    Args:
    messages: list of messages in the format [{"role": "user"|"assistant", "content": <message_content>}]

    Returns:
    dict: the last assistant message in the conversation
    """
    check_messages(messages)
    if (
        len(messages) < 2
    ):  # there should be at least one user message and one assistant message
        return None
    if messages[-1]["role"] == "user":
        messages = messages[:-1]
    last_message = messages[-1]
    if last_message["role"] != "assistant":
        assert (
            last_message["role"] == system_key
        ), "This is a bug, a last message from a checked message must be 'system' or 'assistant' after removing the last 'user' message."
        return None
    return last_message


#### Deprecated ####
def get_messages(text):
    """
    Deprecated. We delegate chat formatting to external libraries.
    """

    pattern = r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>"
    matches = re.findall(pattern, text, re.DOTALL)

    messages = []
    for role, content in matches:
        messages.append(
            {
                "role": role,
                "content": content.strip(),  # Remove any extra whitespace or newlines
            }
        )
    return messages
