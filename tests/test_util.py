from uscrn._util import retry


def test_retry(caplog):
    from requests.exceptions import ConnectionError

    n = 0

    @retry
    def func():
        nonlocal n
        if n < 2:
            n += 1
            raise ConnectionError

    with caplog.at_level("INFO"):
        func()

    for r in caplog.records:
        assert r.getMessage() == "Retrying func in 1 s after connection error"
