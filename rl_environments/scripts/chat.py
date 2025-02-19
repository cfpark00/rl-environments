import argparse
import requests


def compute_reward(url, messages, hidden_params):
    """
    Compute the reward for the entire conversation by sending
    the messages and hidden parameters to the reward endpoint.
    """
    payload = {
        "messages_b": [messages],
        "hidden_params_b": [hidden_params],
    }
    try:
        response = requests.post(f"{url}/get_reward_batched", json=payload)
        if response.status_code == 200:
            reward_resp = response.json()[0]
            print("Reward:", reward_resp)
        else:
            print(
                f"Reward endpoint returned an error {response.status_code}: {response.text}"
            )
    except Exception as e:
        print("An error occurred while computing reward:", e)


def print_message(messages):
    """
    Print the conversation messages.
    """
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        if role == "assistant":
            print("Assistant:", content)
        elif role == "user":
            print("User:", content)
        else:
            print(f"{role.capitalize()}: {content}")

def reset(url):
    data = requests.post(f"{url}/get_data_sample", json={}).json()
    print("=====================================")
    print("Resetting the conversation.")
    messages, hidden_params = data["messages"], data["hidden_params"]
    print("=====================================")
    print("Conversation:")
    print_message(messages)
    return messages, hidden_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive chat with the environment via a server URL."
    )
    parser.add_argument(
        "url", type=str, help="Server URL (e.g., http://127.0.0.1:8000)"
    )
    args = parser.parse_args()
    url = args.url

    print("=====================================")
    print(f"Connecting to environment at {url}")
    print("=====================================")
    print("You are the assistant. Special functions:")
    print("Type 'exit' or 'quit' to stop.")
    print("Type 'reward' to compute the reward from the entire conversation.")
    print("Type 'hidden' to print the hidden parameters.")
    print("Type 'reset' to reset the conversation.")
    # Retrieve the initial sample messages and hidden parameters.
    data = requests.post(f"{url}/get_data_sample", json={}).json()
    messages, hidden_params = data["messages"], data["hidden_params"]

    print("=====================================")
    print("Conversation:")
    print_message(messages)

    while True:
        # The human is the assistant.
        assistant_input = input("Assistant: ").strip()
        lowercased_input = assistant_input.lower()
        if lowercased_input in ["exit", "quit", "reward", "hidden", "reset"]:
            if lowercased_input in ["exit", "quit"]:
                print("Exiting the chat. Goodbye!")
                break
            elif lowercased_input == "reward":
                compute_reward(url, messages, hidden_params)
                continue
            elif lowercased_input == "hidden":  # print hidden params
                print("Hidden Params:", hidden_params)
                continue
            elif lowercased_input == "reset":
                messages, hidden_params= reset(url)
                continue

        # Append the assistant's message.
        messages.append({"role": "assistant", "content": assistant_input})

        # Build the payload for getting the user's response from the environment.
        payload = {
            "messages_b": [messages],
            "hidden_params_b": [hidden_params],
        }

        try:
            response = requests.post(f"{url}/get_env_response_batched", json=payload)
            if response.status_code == 200:
                # The environment's response (simulated as the User's reply) is expected in a batched list.
                env_response_b = response.json()
                user_response = env_response_b[0]
                print("User:", user_response["content"])
                messages.append(user_response)
                if user_response.get("done", False):
                    print("Conversation is done. Final reward:")
                    print("=====================================")
                    compute_reward(url, messages, hidden_params)
                    messages, hidden_params = reset(url)
            else:
                print(
                    f"Server returned an error {response.status_code}: {response.text}"
                )
        except Exception as e:
            print("An error occurred while communicating with the server:", e)
