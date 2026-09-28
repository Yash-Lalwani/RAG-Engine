from rag_engine.cache.keys import answer_key, intent_key, sql_generation_key
from rag_engine.cache.store import CacheStore, track_cache


def test_memory_store_get_set_and_expiry():
    store = CacheStore()
    store.set_many("t", {"a": "1", "b": "2"}, ttl_seconds=60)
    store.set("t", "old", "x", ttl_seconds=-1)
    assert store.get_many("t", ["b", "missing", "a", "old"]) == ["2", None, "1", None]
    assert store.stats()["t"] == {"hits": 2, "misses": 2}


def test_track_cache_counts_only_inside_the_block():
    store = CacheStore()
    store.set("t", "a", "1", ttl_seconds=60)
    store.get("t", "a")
    with track_cache() as counts:
        store.get_many("t", ["a", "nope"])
        store.get("u", "nope")
    store.get("t", "a")
    assert counts == {"t": {"hits": 1, "misses": 1}, "u": {"hits": 0, "misses": 1}}


def test_answer_key_changes_with_version_question_and_options_but_not_option_order():
    base = answer_key("k8s", 3, "q", {"top_k": 5, "hyde": False})
    assert base == answer_key("k8s", 3, " q ", {"hyde": False, "top_k": 5})
    assert base != answer_key("k8s", 4, "q", {"top_k": 5, "hyde": False})
    assert base != answer_key("k8s", 3, "q", {"top_k": 6, "hyde": False})
    assert base != answer_key("k8s", 3, "other", {"top_k": 5, "hyde": False})


def test_intent_and_sql_keys_include_the_allowed_tables_in_any_order():
    assert intent_key("c", ["b", "a"], "q") == intent_key("c", ["a", "b"], "q")
    assert intent_key("c", ["a"], "q") != intent_key("c", ["a", "b"], "q")
    assert sql_generation_key("db", ["a"], "q") != sql_generation_key("db", ["a", "b"], "q")
