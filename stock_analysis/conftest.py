# Root-level test collection config.
# test_all.py is a live-server integration script (raw `requests` calls, no
# pytest test functions) — exclude it from pytest discovery.
collect_ignore_glob = ["test_all.py"]
