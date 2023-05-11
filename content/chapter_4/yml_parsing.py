import yaml


class InvalidResponse(Exception):
    pass


class InvalidItemType(Exception):
    pass


class InvalidItemKeys(Exception):
    pass


class InvalidItemName(Exception):
    pass


class InvalidItemQuantity(Exception):
    pass


class InvalidItemUnit(Exception):
    pass


# Provided schema
schema = """
- item: Apple Slices
  quantity: 5
  unit: pieces
- item: Milk
  quantity: 1
  unit: gallon
- item: Bread
  quantity: 2
  unit: loaves
- item: Eggs
  quantity: 1
  unit: dozen
"""


def validate_response(response, schema):
    # Parse the schema
    schema_parsed = yaml.safe_load(schema)

    # Check if the response is a list
    if not isinstance(response, list):
        raise InvalidResponse("Response is not a list")

    # Check if each item in the list is a dictionary
    for item in response:
        if not isinstance(item, dict):
            raise InvalidItemType("Response item is not a dictionary")

        # Check if each dictionary has the keys "item", "quantity", and "unit"
        if not all(key in item for key in ("item", "quantity", "unit")):
            raise InvalidItemKeys("Response item does not have the correct keys")

        # Check if the values associated with each key are the correct type
        if not isinstance(item["item"], str):
            raise InvalidItemName("Response item name is not a string")
        if not isinstance(item["quantity"], int):
            raise InvalidItemQuantity("Response item quantity is not an integer")
        if not isinstance(item["unit"], str):
            raise InvalidItemUnit("Response item unit is not a string")

        # Check if the values associated with each key are the correct value
        if item["item"] not in [x["item"] for x in schema_parsed]:
            raise InvalidItemName("Response item name is not in schema")
        if item["quantity"] > 10:
            raise InvalidItemQuantity("Response item quantity is greater than 10")
        if item["unit"] not in ["pieces", "dozen"]:
            raise InvalidItemUnit("Response item unit is not pieces or dozen")


# Fake responses
fake_response_1 = """
- item: Apple Slices
  quantity: 5
  unit: pieces
- item: Eggs
  quantity: 2
  unit: dozen
"""

fake_response_2 = """
# Updated yaml list
- item: Apple Slices
  quantity: 5
  unit: pieces
"""

fake_response_3 = """Unmatched"""

# Parse the fake responses
response_1_parsed = yaml.safe_load(fake_response_1)
response_2_parsed = yaml.safe_load(fake_response_2)
response_3_parsed = yaml.safe_load(fake_response_3)

# Validate the responses against the schema
try:
    validate_response(response_1_parsed, schema)
    print("Response 1 is valid")
except Exception as e:
    print("Response 1 is invalid:", str(e))

try:
    validate_response(response_2_parsed, schema)
    print("Response 2 is valid")
except Exception as e:
    print("Response 2 is invalid:", str(e))

try:
    validate_response(response_3_parsed, schema)
    print("Response 3 is valid")
except Exception as e:
    print("Response 3 is invalid:", str(e))
