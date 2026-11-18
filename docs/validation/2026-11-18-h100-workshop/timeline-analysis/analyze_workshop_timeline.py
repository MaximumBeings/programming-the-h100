#!/usr/bin/env python3
"""Read Nsight Systems 2024.6 SQLite exports; report observed interval overlap."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def analyze(path):
    db=sqlite3.connect('file:'+str(path.resolve())+'?mode=ro',uri=True)
    kernels=list(db.execute('SELECT start,end,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL'))
    copies=list(db.execute('SELECT start,end,streamId,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY'))
    db.close()
    # Count intervals with ANY positive-duration intersection, not a sum of
    # activity durations, and require distinct streams for this comparison.
    def intersects(a,b):return a[2]!=b[2] and max(a[0],b[0])<min(a[1],b[1])
    overlapped=sum(any(intersects(k,c) for c in copies) for k in kernels)
    copy_overlaps=sum(any(intersects(c,other) for other in copies) for c in copies)
    return dict(sqlite=path.name,sqlite_sha256=sha(path),kernels=len(kernels),copies=len(copies),
                worker_streams=sorted({k[2] for k in kernels}),
                payload_bytes=sum(c[3] for c in copies),
                kernels_intersecting_other_stream_copy=overlapped,
                copies_intersecting_other_stream_copy=copy_overlaps)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('serial',type=Path);parser.add_argument('overlap',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise SystemExit('Choose a fresh output path.')
    result={'analyzed_utc':datetime.now(timezone.utc).isoformat(),
            'analysis_source_sha256':sha(Path(__file__)),
            'scope':'CPU analysis of captured trace intervals; not an additional GPU run or unprofiled benchmark.',
            'serial':analyze(args.serial),'overlap':analyze(args.overlap)}
    for mode in ('serial','overlap'):
        row=result[mode]
        if row['kernels']!=128 or row['copies']!=256 or row['payload_bytes']!=268435456:
            raise SystemExit('Captured workload counts differ: '+mode)
    if (len(result['serial']['worker_streams'])!=1 or len(result['overlap']['worker_streams'])!=2
            or result['serial']['kernels_intersecting_other_stream_copy']!=0
            or result['overlap']['kernels_intersecting_other_stream_copy']==0):
        raise SystemExit('The captured overlap hypothesis is not established.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    print('TIMELINE ANALYSIS PASS: actual counts and cross-stream interval intersections checked')
if __name__=='__main__':main()
