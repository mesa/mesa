"""Tests for icon cache functionality."""

import numpy as np

from mesa.visualization.icon_cache import IconCache


def test_icon_cache_matplotlib():
    """Test icon cache with matplotlib backend."""
    cache = IconCache(backend="matplotlib")
    result = cache.get_or_create("smiley", size=32)

    # Should return numpy array for matplotlib
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (32, 32, 4)  # RGBA


def test_icon_cache_altair():
    """Test icon cache with altair backend."""
    cache = IconCache(backend="altair")
    result = cache.get_or_create("smiley", size=32)

    # Should return data URL string for altair
    assert result is not None
    assert isinstance(result, str)
    assert result.startswith("data:image/png;base64,")


def test_icon_cache_caches_results():
    """Test that icon cache actually caches and reuses results."""
    cache = IconCache(backend="altair")

    # First call - creates the icon
    result1 = cache.get_or_create("smiley", size=24)
    assert len(cache) == 1

    # Second call - should return cached result
    result2 = cache.get_or_create("smiley", size=24)
    assert len(cache) == 1  # Still only 1 cached item
    assert result1 == result2  # Same result

    # Different size - creates new cache entry
    result3 = cache.get_or_create("smiley", size=48)
    assert len(cache) == 2  # Now 2 cached items
    assert result1 != result3  # Different result


def test_icon_cache_different_icons():
    """Test caching different icon names."""
    cache = IconCache(backend="altair")

    smiley = cache.get_or_create("smiley", size=24)
    sad = cache.get_or_create("sad_face", size=24)
    neutral = cache.get_or_create("neutral_face", size=24)

    assert len(cache) == 3
    assert smiley is not None
    assert sad is not None
    assert neutral is not None
    # Each icon should be different
    assert smiley != sad
    assert sad != neutral


def test_icon_cache_get_returns_none_for_missing():
    """Test that get returns None for uncached icons."""
    cache = IconCache(backend="altair")

    result = cache.get("nonexistent", size=24)
    assert result is None


def test_icon_cache_get_returns_none_for_none_name():
    """Test that get handles None icon name."""
    cache = IconCache(backend="altair")

    result = cache.get(None, size=24)
    assert result is None


def test_icon_cache_clear():
    """Test clearing the cache."""
    cache = IconCache(backend="altair")

    cache.get_or_create("smiley", size=24)
    cache.get_or_create("sad_face", size=24)
    assert len(cache) == 2

    cache.clear()
    assert len(cache) == 0


def test_icon_cache_fallback_on_missing_icon():
    """Test fallback behavior for unknown icon names."""
    cache = IconCache(backend="altair")

    # Unknown icon should still return something (fallback circle)
    result = cache.get_or_create("unknown_icon_name", size=24)
    assert result is not None
    assert isinstance(result, str)
    assert result.startswith("data:image/png;base64,")
