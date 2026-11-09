#!/usr/bin/env python3
"""Execute a frozen clocked graph program; correctness requires separate audit."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from fpga.openeye.clocked_core_compat import gate_unused_pooling_ring,retain_zeros_for_clocked_convolution
RTL=ROOT/'fpga/openeye/rtl'
MODULES=('openeye_graph_engine.sv','openeye_graph_tb.sv','openeye_registered_dma_sender.sv',
    'openeye_dma_output_frame_buffer.sv','openeye_activation_dma_stream.sv','openeye_postops_stream.sv',
    'openeye_postops_elastic.sv','openeye_gap_elastic.sv','openeye_logit_affine_stream.sv','openeye_network_cycle_monitor.sv')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(args):
    root=args.root.resolve()
    if ROOT in root.parents or root==ROOT:raise ValueError('Private run root required')
    prep=json.loads((root/'preparation.json').read_text())
    savable=bool(getattr(args,'savable',False))
    stem_lanes=getattr(args,'stem_channel_lanes',1)
    function_metrics=bool(getattr(args,'function_metrics',False))
    if type(stem_lanes)is not int or stem_lanes not in(1,3)or(stem_lanes!=1 and not prep.get('connected_stem')):
        raise ValueError('Research stem lanes require a connected image-to-logit graph')
    for name,digest in prep['files'].items():
        if Path(name).name!=name or sha(root/name)!=digest:raise ValueError('Changed graph model/program')
    build=root/'clocked-build';build.mkdir()
    work=root/'generated'
    gate_unused_pooling_ring(work)
    if prep.get('optimized'):retain_zeros_for_clocked_convolution(work)
    # Correct signed ReLU is shared by core/bypass and graph post-ops. Frozen
    # generated copies only; the upstream checkout is never edited.
    for name in ('openeye-activation-lane-select.patch','openeye-relu-signed.patch'):
        patch=str(ROOT/'fpga/openeye/patches'/name)
        check=subprocess.run(['git','apply','--check',patch],cwd=work,capture_output=True,text=True)
        if check.returncode==0:
            result=subprocess.run(['git','apply',patch],cwd=work,capture_output=True,text=True)
            (build/(name+'.log')).write_text(result.stdout+result.stderr);result.check_returncode()
        else:
            result=subprocess.run(['git','apply','--reverse','--check',patch],cwd=work,capture_output=True,text=True)
            (build/(name+'.log')).write_text('Already applied; exact reverse check required\n'+result.stdout+result.stderr)
            result.check_returncode()
    sources=list((work/'hdl').rglob('*.v'))+[work/'sim/dma_storage.v']
    include=build/'include';include.mkdir()
    for p in (work/'sim').glob('*.vh'):shutil.copyfile(p,include/p.name)
    modules=list(MODULES)
    if prep.get('connected_stem'):
        modules+=['openeye_stem_frontend.sv','openeye_stem_conv_stream.sv','openeye_stem_conv_channel_parallel.sv','openeye_maxpool3_stream.sv']
    if function_metrics:modules+=['openeye_function_cycle_observer.sv','openeye_postops_cycle_metrics.sv']
    for name in modules:
        if savable and name=='openeye_graph_tb.sv':
            from fpga.openeye.resumable_graph_top import source
            p=build/'openeye_graph_resumable.sv';content=source()
            if function_metrics:
                from fpga.openeye.graph_function_metrics import inject
                content=inject(content,connected_stem=bool(prep.get('connected_stem')),stem_lanes=stem_lanes)
            p.write_text(content);sources.append(p)
            continue
        p=build/name;shutil.copyfile(RTL/name,p);sources.append(p)
        if function_metrics and name=='openeye_graph_tb.sv':
            from fpga.openeye.graph_function_metrics import inject
            p.write_text(inject(p.read_text(),connected_stem=bool(prep.get('connected_stem')),stem_lanes=stem_lanes))
    params=dict(MEMORY_VALUES=prep['memory_values'],INITIAL_VALUES=prep['initial_values'],
        PACKET_WORDS=prep['packet_words'],MAP_LANES=prep['map_lanes'],INSTRUCTIONS=prep['instructions'],
        COEFFICIENTS=prep['coefficient_count'],OUTPUT_BASE=prep['output_base'],OUTPUT_VALUES=prep['output_values'],
        FRAME_CAPACITY=prep['raw_words'],USE_STEM=int(prep.get('connected_stem',False)),OPTIMIZED_CORE=int(prep.get('optimized',False)),
        STEM_CHANNEL_LANES=stem_lanes,FUNCTION_METRICS=int(function_metrics))
    binary=build/'obj/Vopeneye_graph_tb'
    if savable:
        cpp=build/'clocked_resume_main.cpp';shutil.copyfile(RTL/cpp.name,cpp);sources.append(cpp)
    mode=['--cc','--exe','--build','--savable','--top-module','openeye_graph_resumable','-o','Vopeneye_graph_tb']if savable else['--binary','--timing','--top-module','openeye_graph_tb']
    command=['verilator',*mode,'--Mdir',str(build/'obj'),
        '-j','2','-CFLAGS','-O3 -std=c++17','-Wno-fatal','-Wno-BLKANDNBLK','-I'+str(include),*[f'-G{k}={v}'for k,v in params.items()],*map(str,sources)]
    if prep.get('optimized'):command.insert(1,'+define+DOMINO_OPTIMIZED_CORE')
    if prep.get('connected_stem'):command.insert(1,'+define+DOMINO_CONNECTED_STEM')
    compile_inputs=sources+list(include.glob('*.vh'))
    start=time.perf_counter_ns()
    if args.compiled_root:
        previous=args.compiled_root.resolve();identity=json.loads((previous/'clocked-build/identity.json').read_text())
        current={str(p.relative_to(root)):sha(p)for p in compile_inputs}
        if (identity['parameters']!=params or identity['host']!=platform.platform() or set(identity['sources'])!=set(current)):
            raise ValueError('Graph binary reuse requires identical RTL/header/geometry/host')
        normalization=[]
        for name,digest in current.items():
            previous_source=previous/name;current_source=root/name
            if sha(previous_source)!=identity['sources'][name]:raise ValueError('Changed original compile source')
            if digest!=identity['sources'][name]:
                def normalized(path):
                    return b'\n'.join(line for line in path.read_bytes().splitlines()if not line.startswith(b'// Auto-generated from '))
                if normalized(previous_source)!=normalized(current_source):raise ValueError('Changed executable graph source')
                normalization.append(dict(path=name,new_comment_only_sha256=digest,original_sha256=sha(previous_source)))
                shutil.copyfile(previous_source,current_source)
        (build/'comment-only-source-normalization.json').write_text(json.dumps(normalization,indent=2)+'\n')
        source=previous/'clocked-build/obj/Vopeneye_graph_tb'
        if sha(source)!=identity['binary_sha256']:raise ValueError('Changed original graph binary')
        binary.parent.mkdir();shutil.copyfile(source,binary);binary.chmod(0o700)
        (build/'compile.log').write_text('Reused exact local binary after source/header/geometry verification\n')
    else:
        result=subprocess.run(command,capture_output=True,text=True)
        (build/'compile.log').write_text(result.stdout+result.stderr);result.check_returncode()
    identity=dict(openeye_pin='9a742fbd42a55ebfa8fcd39aa41dfa36a367572f',simulator='verilator',host=platform.platform(),
        binary_sha256=sha(binary),sources={str(p.relative_to(root)):sha(p)for p in compile_inputs},
        preparation_sha256=sha(root/'preparation.json'),parameters=params,compile_wall_ns=time.perf_counter_ns()-start,
        runner_sha256=sha(Path(__file__)),scope='clocked graph gate; not full network unless complete source graph is independently audited')
    identity['driver_protocol']='verilated-saveable-low-edge-v1'if savable else'verilated-timing-v1'
    (build/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    if args.compile_only or savable:
        print('GRAPH_COMPILED_ONLY; no DUT execution or numeric PASS',flush=True)
        return
    command=[str(binary),f'+PROGRAM={root/"program.hex"}',f'+PACKETS={root/"packets.hex"}',f'+MAP={root/"maps.hex"}',
        f'+COEFFICIENTS={root/"coefficients.hex"}',f'+INITIAL={root/"initial-memory.hex"}',f'+OUTPUT={root/"actual-output.hex"}',
        f'+RAW_OUTPUT={root/"actual-raw.hex"}',f'+ACCEPTED={root/"actual-accepted.hex"}',f'+FORWARDED={root/"actual-forwarded.hex"}']
    if prep.get('connected_stem'):
        command += [f'+STEM_WEIGHTS={root/"stem-weights.hex"}',f'+STEM_COEFFICIENTS={root/"stem-coefficients.hex"}',f'+STEM_OUTPUT={root/"actual-stem.hex"}']
    start=time.perf_counter_ns()
    with (root/'simulation.log').open('x')as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=18000)
    execution=dict(returncode=result.returncode,host_simulation_wall_ns=time.perf_counter_ns()-start,
        identity_sha256=sha(build/'identity.json'),files={p.name:sha(p)for p in [root/'simulation.log',root/'actual-output.hex',root/'actual-raw.hex',root/'actual-accepted.hex',root/'actual-forwarded.hex',root/'actual-stem.hex']if p.exists()})
    (root/'execution.json').write_text(json.dumps(execution,indent=2)+'\n');result.check_returncode()
    print('GRAPH_PROGRAM_COMPLETED; independent numeric audit required',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--compiled-root',type=Path)
    parser.add_argument('--compile-only',action='store_true')
    parser.add_argument('--savable',action='store_true',help='Build the untimed snapshot driver; execution is managed by the resume runner')
    parser.add_argument('--stem-channel-lanes',type=int,choices=(1,3),default=1,
                        help='Explicit research configuration; default preserves scalar published profiles')
    parser.add_argument('--function-metrics',action='store_true',help='Passive per-function and exclusive edge counters; does not alter the datapath')
    execute(parser.parse_args())
