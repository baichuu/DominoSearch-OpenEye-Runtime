"""Synthetic archive/egress rejection tests, not DUT evidence."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET
import pytest

from fpga.openeye.clocked_release_ingress import extract,safe_name,verify_release_folder
from fpga.openeye.seal_clocked_public_evidence import seal


def archive(path,contents,extra=None):
    manifest=dict(schema_version=1,files={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())for name,data in contents.items()})
    with tarfile.open(path,'w:gz')as tar:
        for name,data in {**contents,'package-manifest.json':json.dumps(manifest).encode()}.items():
            item=tarfile.TarInfo(name);item.size=len(data);tar.addfile(item,io.BytesIO(data))
        if extra is not None:tar.addfile(extra,io.BytesIO(b'x')if extra.isfile()else None)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_extract_exact_members_and_hashes(tmp_path):
    path=tmp_path/'data.tar.gz';digest=archive(path,{'graph/program.hex':b'00\n','gate/preparation.json':b'{}'})
    report=extract(path,tmp_path/'output',digest)
    assert report['schema_version']==1
    assert (tmp_path/'output/graph/program.hex').read_bytes()==b'00\n'
    with pytest.raises(ValueError,match='New private'):extract(path,tmp_path/'output',digest)


@pytest.mark.parametrize('shared',[False,True])
def test_child_release_requires_own_owner_only_acl(tmp_path,monkeypatch,shared):
    from types import SimpleNamespace
    import fpga.openeye.network_checkpoint as checkpoint
    import fpga.openeye.clocked_release_ingress as ingress
    config=tmp_path/'config';config.write_text('[domino-drive]\ntoken = {"access_token":"fabricated-unit-token"}\n')
    roots=[];monkeypatch.setattr(checkpoint,'verify_private_drive',lambda cfg,root,policy:roots.append((root,policy)))
    monkeypatch.setattr(ingress.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=json.dumps([dict(Name='release-r1',ID='fabricated-child')])) )
    permissions=[dict(type='user',role='owner')]+([dict(type='user',role='writer')]if shared else [])
    monkeypatch.setattr(checkpoint,'drive_permissions',lambda *a:permissions)
    if shared:
        with pytest.raises(ValueError,match='ACL differs'):verify_release_folder(config,'release-r1')
    else:verify_release_folder(config,'release-r1')
    assert roots==[('domino-drive:DominoSearch-runtime-private','owner-only')]
    with pytest.raises(ValueError):verify_release_folder(config,'../outside')


@pytest.mark.parametrize('name',['../escape','/absolute','graph/../../escape','graph/.git/config','graph/.ssh/config','graph/.venv/token','logs/private.txt','graph/bad\nname','graph\\escape'])
def test_reject_unsafe_archive_names(name):
    with pytest.raises(ValueError):safe_name(name)


@pytest.mark.parametrize('kind',['symlink','hardlink','duplicate','unbound-manifest','checksum'])
def test_archive_rejects_links_duplicates_unbound_files_and_modified_bytes(tmp_path,kind):
    path=tmp_path/'data.tar.gz';extra=tarfile.TarInfo('graph/extra');extra.size=1
    if kind=='symlink':extra.type=tarfile.SYMTYPE;extra.linkname='/tmp/outside';extra.size=0
    if kind=='hardlink':extra.type=tarfile.LNKTYPE;extra.linkname='graph/program.hex';extra.size=0
    if kind=='duplicate':extra.name='graph/program.hex'
    if kind=='unbound-manifest':extra.name='graph/package-manifest.json'
    digest=archive(path,{'graph/program.hex':b'00\n'},None if kind=='checksum'else extra)
    if kind=='checksum':digest='0'*64
    with pytest.raises(ValueError):extract(path,tmp_path/'output',digest)


def test_egress_ignores_private_files_and_keeps_real_failure(tmp_path):
    root=tmp_path/'input';(root/'network').mkdir(parents=True)
    (root/'private-tensor.bin').write_bytes(b'NEVER PUBLISH')
    (root/'network/audit.json').write_text(json.dumps(dict(result='FAIL_CLOCKED_NETWORK_AUDIT',failure_type='ValueError',full_network_cycles=None,full_network_rtl_pass=False)))
    (root/'network/junit.xml').write_text('<testsuite tests="1" failures="1"><testcase name="PRIVATE VALUE"><failure>SECRET RAW LOG</failure></testcase></testsuite>')
    output=tmp_path/'public';report=seal(root,output,'dense','a'*40,'123')
    assert not report['full_network_rtl_pass']
    assert ET.parse(output/'network/junit.xml').getroot().get('failures')=='1'
    assert not any(b'SECRET'in p.read_bytes()or b'PRIVATE'in p.read_bytes()for p in output.rglob('*')if p.is_file())


def test_missing_egress_evidence_does_not_create_pass(tmp_path):
    root=tmp_path/'empty';root.mkdir()
    assert seal(root,tmp_path/'public','optimized','b'*40,'345')['full_network_rtl_pass']is False


@pytest.mark.parametrize('mode',['free-form','uncertified-cycles','junit-disagreement','tensor','unknown-field'])
def test_reject_private_or_inconsistent_public_claims(tmp_path,mode):
    root=tmp_path/'input';(root/'network').mkdir(parents=True)
    report=dict(result='FAIL_CLOCKED_NETWORK_AUDIT',full_network_rtl_pass=False,full_network_cycles=None)
    if mode=='free-form':report['failure_type']='/private/tensor/path'
    if mode=='uncertified-cycles':report['full_network_cycles']=123
    if mode=='junit-disagreement':(root/'network/junit.xml').write_text('<testsuite tests="1" failures="0"><testcase/></testsuite>')
    if mode=='tensor':report['stages']=[{'weights':[1,2,3]}]
    if mode=='unknown-field':report['private_coefficient']=123
    (root/'network/audit.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):seal(root,tmp_path/'public','mac20','c'*40,'456')
