#!/usr/bin/env python3
"""Verify/extract immutable private inputs for the public clocked-runtime jobs."""
import argparse
import configparser
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

PRIVATE_ROOT='domino-drive:DominoSearch-runtime-private'
PROFILES=('dense','mac20','optimized')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def safe_name(name):
    p=Path(name)
    require(name and '\\'not in name and not p.is_absolute()and '..'not in p.parts and
            all(part not in('.git','.ssh','.venv','__pycache__')for part in p.parts)and
            not any(ord(c)<32 for c in name),'Unsafe archive member path')
    require(p.parts[0]in('graph','gate','bundle','package-manifest.json'),'Unexpected private input category')
    return p


def verify_release_folder(config_path,folder):
    """Read-only owner-only validation of root and exact immutable child."""
    require(re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}',folder)is not None,'Unsafe immutable release folder')
    from fpga.openeye.network_checkpoint import verify_private_drive,drive_permissions,validate_sharing
    verify_private_drive(config_path,PRIVATE_ROOT,'owner-only')
    result=subprocess.run(['rclone','lsjson','--dirs-only','--config',str(config_path),PRIVATE_ROOT],capture_output=True,text=True)
    require(result.returncode==0,'Cannot inspect immutable private release folder')
    rows=json.loads(result.stdout);matches=[v for v in rows if v.get('Name')==folder]
    require(len(matches)==1 and matches[0].get('ID'),'Require unique existing immutable private release folder')
    config=configparser.ConfigParser(interpolation=None);config.read(config_path)
    token=json.loads(config['domino-drive']['token'])['access_token']
    validate_sharing(drive_permissions(matches[0]['ID'],token),'owner-only')


def extract(archive,output,expected_sha256):
    require(re.fullmatch('[0-9a-f]{64}',expected_sha256)is not None and sha(archive)==expected_sha256,'Private archive checksum mismatch')
    require(not output.exists(),'New private extraction path required')
    with tarfile.open(archive,'r:gz')as tar:
        members=tar.getmembers();require(len(members)<30000,'Excessive private archive members')
        seen=set();total=0
        for member in members:
            path=safe_name(member.name);key=str(path)
            require(key not in seen,'Duplicate archive path');seen.add(key)
            require(member.isfile()or member.isdir(),'Links/devices are forbidden in runtime input archives')
            require(member.size>=0,'Negative archive file size');total+=member.size
        require(total<=8*(1<<30),'Private input archive exceeds runner capacity')
        output.mkdir(mode=0o700)
        for member in members:
            path=output/safe_name(member.name)
            if member.isdir():path.mkdir(parents=True,exist_ok=True,mode=0o700)
            else:
                path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
                source=tar.extractfile(member);require(source is not None,'Missing archive stream')
                with path.open('xb')as target:
                    for block in iter(lambda:source.read(1<<20),b''):target.write(block)
                path.chmod(0o600)
    manifest=json.loads((output/'package-manifest.json').read_text())
    require(manifest['schema_version']==1,'Unknown private runtime package schema')
    actual={str(p.relative_to(output))for p in output.rglob('*')if p.is_file()and p!=output/'package-manifest.json'}
    require(actual==set(manifest['files']),'Missing/extra unbound private input files')
    for name,record in manifest['files'].items():
        path=output/safe_name(name)
        require(path.stat().st_size==record['bytes']and sha(path)==record['sha256'],'Private input member identity mismatch')
    return manifest


def ingress(args):
    release=json.loads(args.release.read_text());require(args.profile in PROFILES,'Unknown model/profile')
    require(release['ready_for_dispatch']is True and release['schema_version']==1 and
            release['experiment']=='whole-network-int8-fixedpoint-v1','Clocked comparison release not ready')
    profile=release['profiles'][args.profile]
    require(profile['model']==('dense'if args.profile=='dense'else'mac20')and profile['optimized']==(args.profile=='optimized'),
            'Profile/model optimization identity mismatch')
    if args.archive:archive=args.archive
    else:
        require(args.config is not None,'Existing private Drive config required')
        name=profile['archive_name'];folder=release['private_release_folder']
        require(re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]+',name)is not None,'Unsafe immutable input location')
        verify_release_folder(args.config,folder)
        archive=args.output.parent/(args.profile+'-clocked-inputs.tar.gz')
        require(not archive.exists(),'Never overwrite downloaded private archive')
        result=subprocess.run(['rclone','copyto','--immutable','--config',str(args.config),
            PRIVATE_ROOT+'/'+folder+'/'+name,str(archive)],capture_output=True,text=True)
        require(result.returncode==0,'Private immutable runtime input retrieval failed')
    require(archive.stat().st_size==profile['archive_bytes'],'Private archive byte-size mismatch')
    manifest=extract(archive,args.output,profile['archive_sha256'])
    require(manifest['model']==profile['model']and manifest['optimized']==profile['optimized']and
            manifest['openeye_pin']==release['openeye_pin']and manifest['arithmetic_contract_sha256']==release['arithmetic_contract_sha256'],
            'Private package experiment/source identity mismatch')
    for name,key in(('graph','graph_preparation_sha256'),('gate','gate_preparation_sha256')):
        require(sha(args.output/name/'preparation.json')==profile[key],'Different prepared graph/gate')
    from fpga.openeye.validate_graph_program import validate
    structural=validate(args.output/'graph')
    return dict(result='PASS_IMMUTABLE_INPUT_INGRESS_NOT_DUT_EVIDENCE',profile=args.profile,model=profile['model'],
        optimized=profile['optimized'],archive_sha256=profile['archive_sha256'],
        graph_preparation_sha256=profile['graph_preparation_sha256'],gate_preparation_sha256=profile['gate_preparation_sha256'],
        commands=structural['commands'],openeye_pin=release['openeye_pin'],
        full_network_cycles=None,full_network_rtl_pass=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--profile',choices=PROFILES,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--config',type=Path);parser.add_argument('--archive',type=Path)
    args=parser.parse_args();report=ingress(args)
    (args.output/'ingress-public.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified immutable private graph input; no DUT result claimed')
