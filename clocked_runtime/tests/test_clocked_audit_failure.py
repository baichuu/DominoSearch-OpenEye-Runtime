"""Numeric audit diagnostics disclose source coordinates, never tensor values."""
import json
import pytest
from fpga.openeye.audit_clocked_graph import ClockedAuditFailure,check_words,failure_report
from fpga.openeye.seal_clocked_public_evidence import seal


def test_equal_word_stream_has_no_failure():
    check_words([0,2**63,9],[0,2**63,9],'RAW_OUTPUT_MISMATCH',15,100)


@pytest.mark.parametrize('actual,expected,offset', [([10,987654321],[10,123456789],101),([10],[10,11],101)])
def test_failure_coordinates_never_disclose_private_values(actual,expected,offset):
    with pytest.raises(ClockedAuditFailure)as caught:
        check_words(actual,expected,'RAW_OUTPUT_MISMATCH',15,100)
    report=failure_report(caught.value)
    assert report['failure_instruction']==15 and report['failure_word_offset']==offset
    assert report['full_network_cycles']is None and report['full_network_rtl_pass']is False
    assert '987654321'not in json.dumps(report)and '123456789'not in str(caught.value)


def test_unclassified_error_body_stays_private():
    report=failure_report(ValueError('/private/tensor coefficient=987654321'))
    assert '/private'not in json.dumps(report)and '987654321'not in json.dumps(report)
    assert 'failure_code'not in report


@pytest.mark.parametrize('code',['RAW_OUTPUT_MISMATCH','PRIVATE_WORD_987654321'])
def test_public_artifact_keeps_only_known_failure_codes(tmp_path,code):
    root=tmp_path/'private';(root/'network').mkdir(parents=True)
    report=failure_report(ClockedAuditFailure('RAW_OUTPUT_MISMATCH',15,100));report['failure_code']=code
    (root/'network/audit.json').write_text(json.dumps(report))
    (root/'network/junit.xml').write_text('<testsuite tests="1" failures="1"><testcase><failure/></testcase></testsuite>')
    if code=='RAW_OUTPUT_MISMATCH':
        sealed=seal(root,tmp_path/'public','optimized','a'*40,'123')
        assert not sealed['full_network_rtl_pass']
        assert json.loads((tmp_path/'public/network/audit.json').read_text())==report
    else:
        with pytest.raises(ValueError,match='Unallowlisted'):seal(root,tmp_path/'public','optimized','a'*40,'123')
