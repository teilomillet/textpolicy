import pytest


def test_validate_installation_runs():
    from textpolicy.validate import validate_installation

    report = validate_installation(verbose=False)

    assert isinstance(report, dict)
    assert 'status' in report
    assert report['status'] in {'ok', 'fail'}
    assert 'checks' in report and isinstance(report['checks'], dict)
