import unittest
from enum import Enum
from typing import List

from nimbusagent.functions.parser import (
    type_mapping,
    extract_params,
    param_to_title,
    extract_enum_values,
    extract_description,
    # import other methods to tests
)


class TestTypeMapping(unittest.TestCase):

    def test_type_mapping_for_basic_types(self):
        self.assertEqual(type_mapping(int), (None, "integer"))
        self.assertEqual(type_mapping(str), (None, "string"))

    def test_type_mapping_for_list(self):
        self.assertEqual(type_mapping(List[int]), ("array", "integer"))
        self.assertEqual(type_mapping(List[str]), ("array", "string"))

    def test_type_mapping_for_unrecognized_types(self):
        self.assertEqual(type_mapping(None), (None, "string"))


class TestExtractParams(unittest.TestCase):

    def test_extract_params_from_docstring(self):
        docstring = '''
        :param arg1: description 1
        :param arg2: description 2
        :return: None
        '''
        self.assertEqual(extract_params(docstring), {'arg1': 'description 1', 'arg2': 'description 2'})


class TestParamToTitle(unittest.TestCase):

    def test_param_to_title(self):
        self.assertEqual(param_to_title("param_name"), "Param Name")


class TestExtractEnumValues(unittest.TestCase):

    def test_extract_enum_values_for_enum(self):
        class MyEnum(Enum):
            A = "A"
            B = "B"

        self.assertEqual(extract_enum_values(MyEnum), ['A', 'B'])


class TestExtractDescription(unittest.TestCase):

    def test_extract_description(self):
        docstring = "Line 1\nLine 2\n:param x: An integer"
        self.assertEqual(extract_description(docstring), "Line 1 Line 2")


# ... Similarly, you can write unittests for func_metadata, build_params, determine_required_parameters ...

if __name__ == "__main__":
    unittest.main()
