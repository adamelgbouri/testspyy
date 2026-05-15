"""
Streamlit cache helpers.

Wraps st.cache_data with sensible defaults so importing modules don't all
have to repeat the boilerplate.
"""

from __future__ import annotations

from functools import wraps
from typing import Callable

import streamlit as st


def cache_data(ttl_seconds: int = 3600):
    """Decorator factory: cache return values for ttl_seconds."""

    def deco(fn: Callable):
        cached = st.cache_data(ttl=ttl_seconds, show_spinner=False)(fn)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return cached(*args, **kwargs)

        return wrapper

    return deco
