"""Sphinx configuration
"""

import dataclasses
import inspect
import operator
import os
import sys
import traceback

sys.path.insert(0, os.path.abspath("../src/"))

project = "SGN-LIGO"
author = "Chad Hanna, Patrick Godwin, Jameson Rollins, James Kennington"
copyright = "2024, IGWN"
_repo_root_url = "https://git.ligo.org/greg/sgn-ligo"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.linkcode",
    "myst_parser",
]

html_theme = "furo"
html_title = "SGN-LIGO"


def linkcode_resolve(domain, info) -> str:
    """Function to resolve links to source code for use with the linkcode extension.

    Args:
        domain:
        info:

    References:
        [1] LinkCode Extension: https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html
        [2] scikit-learn linkcode_resolve: https://github.com/scikit-learn/scikit-learn/blob/f7026e575a494b
                                           47b50557932b5c3ce0688fdc72/doc/sphinxext/github_link.py
        [3] Issue with DataClass attrs: https://stackoverflow.com/questions/77848299/operator-attrgetter-
                                        cannot-read-attributes-in-a-dataclass

    Returns:
        str: URL to source code
    """
    if domain not in ("py", "pyx"):
        return
    if not info.get("module") or not info.get("fullname"):
        return

    try:
        obj_tokens = info["fullname"].split(".")
        class_name = obj_tokens[0]
        module = __import__(info["module"], fromlist=[class_name])

        # Check for dataclass, non-defaulted fields will not be class attrs and will not be importable
        # directly from the type, so we need to get the class object and check the fields
        class_obj = getattr(module, class_name)
        if dataclasses.is_dataclass(class_obj) and len(obj_tokens) > 1:
            fields = dataclasses.fields(class_obj)
            field_dict = {field.name: field for field in fields}
            field = field_dict.get(info["fullname"].split(".")[1])

            # If the field has no default, we can only locate the class, otherwise we can locate the field
            if field is None or field.default is dataclasses.MISSING:
                obj = class_obj
            else:
                obj = operator.attrgetter(info["fullname"])(module)
        else:
            obj = operator.attrgetter(info["fullname"])(module)

        # Unwrap the object to get the correct source
        # file in case that is wrapped by a decorator
        obj = inspect.unwrap(obj)
    except Exception as e:
        # Print formatted traceback, used for debugging
        formatted_traceback = "".join(
            traceback.format_exception(type(e), e, e.__traceback__)
        )
        print(formatted_traceback)
        return

    # Extract the filename
    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return

    fn = os.path.relpath(fn, start=os.path.dirname(__import__("sgn").__file__))

    # Try to get line number
    try:
        lineno = inspect.getsourcelines(obj)[1]
    except Exception:
        lineno = ""

    if lineno:
        lineno = f"#L{lineno}"

    return f"{_repo_root_url}/-/blob/main/src/sgnligo/{fn}{lineno}"
