#!/usr/bin/env python3
"""Readable per-site tables from a completed independent cycle audit only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fpga.openeye.audit_function_metrics import FUNCTION_COLUMNS,STEP_COLUMNS
from fpga.openeye.validate_graph_program import require

GROUPS={
    'image_ingress':('image_ingress',),
    'stem_compute_overlap':('stem_compute_with_overlap',),
    'stem_fill_drain':('stem_fill_drain',),
    'dma_feed_wait':('dma_start','dma_feed','dma_wait'),
    'psum_scatter_accumulate':('scatter','acc_read','acc_write'),
    'sram_read_wait':('scan_read','scan_wait','skip_read','skip_wait'),
    'postops_send_wait':('post_send','post_wait'),
    'gap_send_wait':('gap_send','gap_wait'),
    'fc_affine_send_wait':('logit_send','logit_wait'),
    'sram_store':('store',),
    'controller':('fetch','next','graph_launch'),
}


def summarize(report):
    require(report.get('result')=='PASS_INDEPENDENT_COMPLETE_CLOCKED_NETWORK_AUDIT'and
            report.get('full_network_rtl_pass')is True and
            report.get('function_metrics_protocol')=='passive-function-cycles-v1',
            'A completed independent network function audit is required')
    functions=report['functions'];steps=report['exclusive_steps'];total=report['full_network_cycles']
    require(type(total)is int and total>0 and total==report['exclusive_cycles_sum']and
            sum(row['cycles']for row in steps)==total,'Exclusive cycle table does not close')
    require(functions and all(set(row)==set(FUNCTION_COLUMNS)and row['additive_to_network']is False for row in functions)and
            all(set(row)==set(STEP_COLUMNS)and row['additive_to_network']is True and
                type(row['cycles'])is int and row['cycles']>=0 for row in steps),'Unknown or overlapping table schema')
    reverse={step:group for group,states in GROUPS.items()for step in states};sites={}
    for row in steps:
        require(row['step']in reverse,'Unclassified measured controller state')
        site=row['site']
        sites.setdefault(site,dict(site=site,**{name:0 for name in GROUPS},total_cycles=0))
        sites[site][reverse[row['step']]]+=row['cycles'];sites[site]['total_cycles']+=row['cycles']
    rows=[]
    for site in ['stem']+[name for name in sites if name not in('stem','dispatch')]+['dispatch']:
        if site in sites:rows.append(sites[site])
    require(sum(row['total_cycles']for row in rows)==total,'Missing full-network site')
    return rows


def write(audit,output):
    report=json.loads(audit.read_text());rows=summarize(report)
    require(not output.exists(),'New table directory required');output.mkdir(parents=True)
    def csv_table(name,values,columns):
        with(output/name).open('w',newline='')as stream:
            writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader();writer.writerows(values)
    csv_table('full-network-by-site.csv',rows,list(rows[0]))
    csv_table('functions.csv',report['functions'],FUNCTION_COLUMNS)
    csv_table('exclusive-steps.csv',report['exclusive_steps'],STEP_COLUMNS)
    lines=['# Cycle của từng bước trong full network','',
        f"Model: `{report['model']}`; optimized: `{report['optimized']}`; stem lanes: {report['stem_channel_lanes']}.",
        f"**Tổng đo từ RTL: {report['full_network_cycles']:,} cycles.** Tổng các dòng exclusive khớp chính xác.",
        '', 'Một verification vector INT8/fixed-point; không phải ImageNet accuracy hoặc latency đo trên board.',
        'DMA gồm cấp dữ liệu và chờ core. SRAM/PSUM/controller được tính riêng. Stem compute chạy chồng lấp BN/ReLU/maxpool.',
        '', '| Site | Tổng cycle | DMA | PSUM | SRAM read/wait + store | Postops | GAP | FC affine | Controller | Stem/input |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    if report.get('postops_pipelined'):
        lines[4:4]=['Postops controller chạy overlap: các lần ghi SRAM diễn ra trong slot SCAN_WAIT/SKIP_WAIT/POST_SEND/POST_WAIT.',
            'Cột SRAM store chỉ đếm cycle riêng ở state STORE; giá trị 0 không có nghĩa là không ghi dữ liệu. Số output writes vẫn được kiểm đủ.', '']
    for row in rows:
        values=[row['total_cycles'],row['dma_feed_wait'],row['psum_scatter_accumulate'],row['sram_read_wait']+row['sram_store'],
                row['postops_send_wait'],row['gap_send_wait'],row['fc_affine_send_wait'],row['controller'],
                row['image_ingress']+row['stem_compute_overlap']+row['stem_fill_drain']]
        lines.append('| '+row['site']+' | '+' | '.join(f'{value:,}'for value in values)+' |')
    lines+=['','## Latency qua từng function','',
        '**Không cộng bảng này vào tổng phía trên:** các function trong pipeline có thể chạy chồng lấp.',
        '`min–max` là cycle mỗi transaction: mỗi value cho affine/ReLU/FC, mỗi channel cho GAP, cả tensor cho stem/maxpool/ingress.',
        '`span` đi từ input đầu đến output cuối của function ở site đó, bao gồm khoảng trống giữa các value.',
        'BN affine đã fold scale; residual scale + add và FC scale + bias là các phép fused, không có register để tách cycle sâu hơn.',
        'ReLU bị tắt được ghi NOT_EXECUTED, không được diễn giải là ReLU có latency 0.',
        '', '| Site | Function | Transactions | Unit | Cycle min–max | Tensor span | Status |',
        '|---|---|---:|---|---:|---:|---|']
    for row in report['functions']:
        lines.append(f"| {row['site']} | {row['function']} | {row['transactions']:,} | {row['latency_unit']} | "
            f"{row['latency_min_cycles']:,}–{row['latency_max_cycles']:,} | {row['tensor_span_cycles']:,} | {row['status']} |")
    (output/'tables.md').write_text('\n'.join(lines)+'\n')
    manifest=dict(protocol='passive-function-cycles-v1',source_audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
        binary_sha256=report['binary_sha256'],preparation_sha256=report['preparation_sha256'],
        full_network_cycles=report['full_network_cycles'],exclusive_cycles_sum=sum(row['total_cycles']for row in rows),
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in output.iterdir()})
    (output/'table-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(write(args.audit,args.output),indent=2))
