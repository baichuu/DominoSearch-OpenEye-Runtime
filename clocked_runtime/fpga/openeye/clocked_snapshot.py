"""Authenticated private model/trace checkpoints; never public artifacts."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import tarfile
import tempfile

from fpga.openeye.clocked_resume import canonical,sha,validate_chain,TRACE_NAMES
from fpga.openeye.validate_graph_program import require

MAGIC=b'DOMINO-CLOCKED-SNAPSHOT-v1\0';MAX_BYTES=4*(1<<30)


def safe_path(name):
    p=Path(name)
    require(name and not p.is_absolute()and '..'not in p.parts and '\\'not in name and
            all(part not in('.git','.ssh','.venv','__pycache__')for part in p.parts),'Unsafe snapshot path')
    simple=set(TRACE_NAMES)|{'resume-chain.json','resume-status.json','execution.json','gate-audit.json','gate-junit.xml','checked-prefixes.json','snapshot-package.json'}
    valid=(name in simple or name=='clocked-build/obj/Vopeneye_graph_tb'or name=='clocked-build/identity.json'or
           re.fullmatch(r'resume-chunks/chunk-\d{4}\.model',name)or
           len(p.parts)>1 and p.parts[0]in('generated','clocked-build')and p.suffix in('.v','.vh','.sv','.cpp'))
    require(bool(valid),'Unallowlisted private snapshot member');return p


def encrypt(source,destination,context,key):
    from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
    require(len(key)==32 and not destination.exists(),'Fresh encrypted snapshot and existing32-byte key required')
    aad=canonical(context);require(len(aad)<65536,'Oversized snapshot identity')
    nonce=os.urandom(12);cipher=Cipher(algorithms.AES(key),modes.GCM(nonce)).encryptor();cipher.authenticate_additional_data(aad)
    with source.open('rb')as stream,os.fdopen(os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb')as out:
        out.write(MAGIC+struct.pack('>I',len(aad))+aad+nonce)
        for block in iter(lambda:stream.read(1<<20),b''):out.write(cipher.update(block))
        out.write(cipher.finalize());out.write(cipher.tag)


def decrypt(source,destination,context,key):
    from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
    from cryptography.exceptions import InvalidTag
    require(source.is_file()and not source.is_symlink()and source.stat().st_size<MAX_BYTES and not destination.exists(),'Unsafe encrypted snapshot')
    try:
        with source.open('rb')as stream:
            require(stream.read(len(MAGIC))==MAGIC,'Unknown encrypted snapshot schema')
            raw=stream.read(4);require(len(raw)==4,'Truncated snapshot header');length=struct.unpack('>I',raw)[0]
            require(0<length<65536,'Invalid snapshot identity length');aad=stream.read(length)
            require(aad==canonical(context),'Different checkpoint execution, profile, pin or input source')
            nonce=stream.read(12);require(len(nonce)==12,'Missing snapshot nonce')
            remaining=source.stat().st_size-stream.tell()-16;require(remaining>0,'Truncated snapshot authentication tag')
            stream.seek(-16,2);tag=stream.read(16);stream.seek(len(MAGIC)+4+length+12)
            cipher=Cipher(algorithms.AES(key),modes.GCM(nonce,tag)).decryptor();cipher.authenticate_additional_data(aad)
            with os.fdopen(os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb')as out:
                while remaining:
                    block=stream.read(min(remaining,1<<20));require(bool(block),'Truncated encrypted model');remaining-=len(block)
                    out.write(cipher.update(block))
                out.write(cipher.finalize())
    except (InvalidTag,ValueError):
        destination.unlink(missing_ok=True)
        raise ValueError('Private checkpoint authentication or execution identity failed')from None


def members(root):
    chain=validate_chain(root);build=json.loads((root/'clocked-build/identity.json').read_text())
    names=set(TRACE_NAMES)|{'resume-chain.json','resume-status.json','execution.json','clocked-build/identity.json','clocked-build/obj/Vopeneye_graph_tb','gate-audit.json','gate-junit.xml','checked-prefixes.json'}|set(build['sources'])
    if chain['chunks'][-1]['snapshot']:names.add(chain['chunks'][-1]['snapshot'])
    for name in names:
        safe_path(name);require((root/name).is_file()and not(root/name).is_symlink(),'Missing actual checkpoint member')
    return sorted(names)


def seal(root,destination,context,key):
    require(not destination.exists(),'Never overwrite an encrypted checkpoint');names=members(root)
    manifest=dict(schema_version=1,context=context,files={n:dict(bytes=(root/n).stat().st_size,sha256=sha(root/n))for n in names})
    require(sum(v['bytes']for v in manifest['files'].values())<MAX_BYTES,'Snapshot exceeds private checkpoint capacity')
    with tempfile.TemporaryDirectory(prefix='clocked-seal-',dir=destination.parent)as tmp:
        archive=Path(tmp)/'payload.tar.gz'
        with tarfile.open(archive,'w:gz',compresslevel=1)as tar:
            for name in names:tar.add(root/name,arcname=name,recursive=False)
            data=canonical(manifest);entry=tarfile.TarInfo('snapshot-package.json');entry.size=len(data);entry.mode=0o600;tar.addfile(entry,io.BytesIO(data))
        encrypt(archive,destination,context,key)
    return sha(destination)


def restore(source,root,context,key):
    require(not(root/'resume-chain.json').exists(),'Restore only into fresh authenticated input copies')
    with tempfile.TemporaryDirectory(prefix='clocked-restore-',dir=root.parent)as tmp:
        staging=Path(tmp);archive=staging/'authenticated.tar.gz';decrypt(source,archive,context,key)
        with tarfile.open(archive,'r:gz')as tar:
            records=tar.getmembers();require(len(records)<1000 and sum(v.size for v in records)<MAX_BYTES,'Oversized snapshot archive')
            names=[str(safe_path(v.name))for v in records]
            require(len(names)==len(set(names))and all(v.isfile()for v in records),'Duplicate/link/device in private checkpoint')
            item=tar.getmember('snapshot-package.json');manifest=json.load(tar.extractfile(item))
            require(manifest['schema_version']==1 and manifest['context']==context and set(names)==set(manifest['files'])|{'snapshot-package.json'},'Unbound checkpoint payload')
            for item in records:
                path=staging/'files'/safe_path(item.name);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
                with path.open('xb')as out:
                    stream=tar.extractfile(item)
                    for block in iter(lambda:stream.read(1<<20),b''):out.write(block)
                if item.name!='snapshot-package.json':
                    expected=manifest['files'][item.name]
                    require(path.stat().st_size==expected['bytes']and sha(path)==expected['sha256'],'Changed private snapshot member')
            build=json.loads((staging/'files/clocked-build/identity.json').read_text())
            require(build['preparation_sha256']==sha(root/'preparation.json')and
                    build['binary_sha256']==manifest['files']['clocked-build/obj/Vopeneye_graph_tb']['sha256'],'Checkpoint belongs to different model or binary')
            if 'stem_channel_lanes'in context:
                require(build['parameters'].get('STEM_CHANNEL_LANES',1)==context['stem_channel_lanes'],
                        'Authenticated checkpoint contains the wrong stem hardware policy')
            if 'function_metrics'in context:
                require(build['parameters'].get('FUNCTION_METRICS',0)==context['function_metrics'],
                        'Authenticated checkpoint contains the wrong function counter policy')
            if 'postops_pipelined'in context:
                require(build['parameters'].get('POSTOPS_PIPELINED',0)==context['postops_pipelined'],
                        'Authenticated checkpoint contains the wrong postops controller policy')
            for name in manifest['files']:
                target=root/safe_path(name);target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
                require(not target.is_symlink()and(not target.exists()or Path(name).parts[0]=='generated'),'Do not overwrite existing execution evidence')
                (staging/'files'/name).replace(target);target.chmod(0o700 if name.endswith('/Vopeneye_graph_tb')else 0o600)
    return validate_chain(root)
