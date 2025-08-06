from enum import Enum
import pytest
from nimbusagent.functions.parser import (
    type_mapping,
    extract_params,
    param_to_title,
    extract_enum_values,
    extract_description,
    # import other methods to tests
)


class TestTypeMapping:

    @pytest.mark.parametrize(
        "input_type, expected",
        [
            (int, (None, "integer")),
            (str, (None, "string")),
            (list[int], ("array", "integer")),
            (list[str], ("array", "string")),
            (None, (None, "string")),
        ],
    )
    def test_type_mapping_for_basic_types(self, input_type, expected):
        assert type_mapping(input_type) == expected


class TestExtractParams:

    def test_extract_params_from_docstring(self):
        docstring = """
        :param arg1: description 1
        :param arg2: description 2
        :return: None
        """
        assert extract_params(docstring) == {
            "arg1": "description 1",
            "arg2": "description 2",
        }


class TestParamToTitle:

    def test_param_to_title(self):
        assert param_to_title("param_name") == "Param Name"


class TestExtractEnumValues:

    def test_extract_enum_values_for_enum(self):
        class MyEnum(Enum):
            A = "A"
            B = "B"

        assert extract_enum_values(MyEnum) == ["A", "B"]


class TestExtractDescription:

    def test_extract_description(self):
        docstring = "Line 1\nLine 2\n:param x: An integer"
        assert extract_description(docstring) == "Line 1 Line 2"


# ... Similarly, you can write unittests for func_metadata, build_params, determine_required_parameters ...
