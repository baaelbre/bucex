import bucex as bx


def test_txm_and_tnm_are_bundled_gaussian_series():
    assert bx.UCCLE_INFO["TXm"]["family"] == "gaussian"
    assert bx.UCCLE_INFO["TNm"]["family"] == "gaussian"
    for name in ("TXm", "TNm"):
        values = bx.load_uccle_series(name, start="2020-01-01", end="2020-12-31")
        assert len(values) == 12
        assert values.notna().all()
