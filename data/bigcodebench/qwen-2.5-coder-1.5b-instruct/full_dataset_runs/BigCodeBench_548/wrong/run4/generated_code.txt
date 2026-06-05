import random
import string
import base64
import zlib
def task_func(string_length=100):
    """
    Create a random string of a specified length with uppercase letters and digits, compress it with zlib, 
    and then encode the compressed string in base64.

    Parameters:
    - string_length (int, optional): The length of the random string to be generated. Default is 100.

    Returns:
    str: The compressed string in base64.

    Requirements:
    - base64
    - zlib
    - random
    - string

    Example:
    >>> random.seed(1)
    >>> compressed_string = task_func(50)
    >>> print(compressed_string)
    eJxzNTH0CgqMMHJxMgkwdAyM8rQwc3IMMffzCHDyCAjy9PQI9HY0CY1wtzRx9YmKMg8wjgQAWN0NxA==
    """
    random_string = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(string_length))
    compressed_string = zlib.compress(random_string.encode())
    return base64.b64encode(compressed_string).decode()