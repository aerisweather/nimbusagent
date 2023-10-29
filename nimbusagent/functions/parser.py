# credit to https://github.com/aurelio-labs/funkagent
import functools
import inspect
import re
import typing


def type_mapping(dtype):
    type_map = {
        float: "number",
        int: "integer",
        str: "string"
    }

    # Check if it's a List from the typing module
    if hasattr(dtype, '__origin__') and dtype.__origin__ == list:
        # Assuming only one argument for List (like List[int]), else default to string
        item_type = dtype.__args__[0] if dtype.__args__ else str
        return "array", type_map.get(item_type, "string")

    return None, type_map.get(dtype, "string")


def extract_params(doc_str: str):
    # split doc string by newline, skipping empty lines
    params_str = [line for line in doc_str.split("\n") if line.strip()]

    params = {}
    current_param_name = None
    current_param_desc = []

    for line in params_str:
        # If we encounter a return annotation, we end processing for the current parameter
        if line.strip().startswith(':return:'):
            if current_param_name:
                params[current_param_name] = ' '.join(current_param_desc).strip()
                current_param_name = None
                current_param_desc = []
            continue

        # we only look at lines starting with ':param'
        if line.strip().startswith(':param'):
            # Save the previous parameter and its description (if any)
            if current_param_name:
                params[current_param_name] = ' '.join(current_param_desc).strip()
                current_param_desc = []

            param_match = re.findall(r'(?<=:param )\w+', line)
            if param_match:
                current_param_name = param_match[0]
                desc_match = line.replace(f":param {current_param_name}:", "").strip()
                # if there is a description, store it
                if desc_match:
                    current_param_desc.append(desc_match)
        elif current_param_name:  # We're collecting multiline descriptions
            current_param_desc.append(line.strip())

    # Save the last parameter and its description (if any)
    if current_param_name:
        params[current_param_name] = ' '.join(current_param_desc).strip()

    return params


def param_to_title(param_name: str) -> str:
    """Converts a parameter name to title format."""
    return param_name.replace('_', ' ').title()


def extract_enum_values(dtype) -> typing.Optional[typing.List[str]]:
    # Check if it's an Enum type
    if dtype.__class__.__name__ == 'EnumMeta':
        return [e.value for e in dtype]

        # Check if it's a typing.Literal type (or similar constructs in the typing module)
    elif hasattr(dtype, '__origin__') and dtype.__origin__ == typing.Literal:
        return list(dtype.__args__)

        # Adjusting the check here for Python 3.8:
    elif getattr(dtype, '__origin__', None) == list:
        inner_dtype = dtype.__args__[0] if hasattr(dtype, '__args__') else None

        # For this example, I'm just checking if it's an Enum or a Literal type inside a List.
        if inner_dtype and inner_dtype.__class__.__name__ == 'EnumMeta':
            return [e.value for e in inner_dtype]
        elif hasattr(inner_dtype, '__origin__') and inner_dtype.__origin__ == typing.Literal:
            return list(inner_dtype.__args__)

    return None


def extract_description(func_doc):
    lines = func_doc.split("\n")
    description_lines = []

    for line in lines:
        if line.strip().startswith(":"):
            break
        description_lines.append(line.strip())

    return ' '.join(description_lines).strip()


def func_metadata(obj):
    """Extracts metadata from a function or a class with a call method."""

    if isinstance(obj, type):  # If it's a class
        method_to_use = getattr(obj, 'method_name', 'call')
        func = getattr(obj, method_to_use, None)
        if not callable(func):
            raise ValueError(f"Class {obj.__name__} does not have a callable 'call' method.")
        func_name = obj.__name__  # Use the class name
    else:  # If it's a function
        func = obj
        func_name = func.__name__  # Use the function name

    # Handle functools.partial and functools.partialmethod
    fixed_args = {}
    if isinstance(func, (functools.partial, functools.partialmethod)):
        fixed_args = func.keywords or {}
        _func = func.func
        if not fixed_args:
            # noinspection PyUnresolvedReferences
            fixed_args = dict(zip(_func.__code__.co_varnames, func.args))
    else:
        _func = func

    argspec = inspect.getfullargspec(_func)
    func_doc = inspect.getdoc(_func)
    func_description = extract_description(func_doc)
    param_details = extract_params(func_doc) if func_doc else {}

    params = build_params(argspec, fixed_args, param_details)

    _required = determine_required_parameters(argspec, fixed_args)

    return {
        "name": func_name,  # This now correctly uses either the class name or the function name
        "description": func_description,
        "parameters": {
            "type": "object",
            "properties": params,
            "required": _required
        }
    }


def build_params(argspec, fixed_args, param_details):
    """Builds parameters metadata."""
    params = {}
    for param_name, param_annotation in argspec.annotations.items():
        if param_name not in fixed_args.keys() and param_name != 'self':
            container_type, mapped_type = type_mapping(param_annotation)

            param_metadata = {
                "title": param_details.get(f"{param_name}_title", param_to_title(param_name)),
                "description": param_details.get(param_name) or "",
                "type": mapped_type
            }

            if container_type == "array":
                param_metadata["type"] = container_type
                param_metadata["items"] = {"type": mapped_type}

            enum_values = extract_enum_values(param_annotation)
            if enum_values:
                param_metadata["enum"] = enum_values

            params[param_name] = param_metadata

    return params


def determine_required_parameters(argspec, fixed_args):
    """Determines the required parameters."""
    _required = [i for i in argspec.args if i not in fixed_args.keys() and i != 'self']

    if argspec.defaults:
        num_defaults = len(argspec.defaults)
        args_with_defaults = argspec.args[-num_defaults:]  # Get the names of arguments with defaults

        _required = [arg for arg in _required if arg not in args_with_defaults]  # Filter out args with defaults

    return _required
