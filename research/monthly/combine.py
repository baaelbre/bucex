"""Merge disjoint compact monthly exports and build one overview."""
import argparse
from pathlib import Path, PurePosixPath
import zipfile
from research.monthly.study_report import build


def combine(archives,output,*,tier='screen'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for source in archives:
        with zipfile.ZipFile(source) as z:
            for entry in z.infolist():
                if entry.is_dir():continue
                name=PurePosixPath(entry.filename)
                if name.is_absolute() or '..' in name.parts or not name.parts or name.parts[0]!=tier:
                    raise ValueError('Unexpected archive member: '+entry.filename)
                # Derived host overviews are regenerated after combining.
                if len(name.parts)>1 and name.parts[1] in ('collected','submissions','plan','bundles','startup_probes','monthly_prior_checks'):
                    continue
                if entry.file_size>100*1024**2:raise ValueError('Expected compact evidence, not fit archives.')
                path=output.joinpath(*name.parts);payload=z.read(entry)
                if path.exists():
                    if path.read_bytes()!=payload:raise ValueError('Conflicting evidence: '+str(path))
                    continue
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    return build(output,tier=tier,batch='monthly_all')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archives',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tier',choices=('screen','paper'),default='screen')
    a=p.parse_args();print(combine(a.archives,a.output,tier=a.tier))
