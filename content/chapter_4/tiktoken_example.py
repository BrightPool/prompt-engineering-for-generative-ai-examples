# 1. Import the package:
import tiktoken

# 2. Load an encoding with tiktoken.get_encoding()
encoding = tiktoken.get_encoding("cl100k_base")

# 3. Turn some text into tokens with encoding.encode()
print(encoding.encode("Learning how to use Tiktoken is fun!"))
print(encoding.decode([1061, 15009, 374, 264, 2294, 1648, 311, 4048, 922, 15592, 0]))


def count_tokens(text_string: str, encoding_name: str) -> int:
    """
    Returns the number of tokens in a text string using a given encoding.

    Args:
        text: The text string to be tokenized.
        encoding_name: The name of the encoding to be used for tokenization.

    Returns:
        The number of tokens in the text string.

    Raises:
        ValueError: If the encoding name is not recognized.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text_string))
    return num_tokens


# 4. Use the function to count the number of tokens in a text string.
text_string = "Hello world! This is a test."
print(count_tokens(text_string, "cl100k_base"))
