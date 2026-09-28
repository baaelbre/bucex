"""Export compact evidence for review, leaving posterior state archives on the host."""
import argparse
from pathlib import Path
import zipfile
from research.seasonal.jobs import ROOT


def export(root,tier,output):
    source=Path(root).resolve()/tier;output=Path(output).resolve()
    if not source.is_dir():raise ValueError(f'Missing results tier: {source}')
    if output.exists():raise FileExistsError(f'{output} exists; choose a new name to preserve that archive')
    allowed={'.csv','.json','.log','.txt','.gz'}
    files=sorted(p for p in source.rglob('*') if p.is_file() and p.suffix in allowed and
                 'attempts' not in p.relative_to(source).parts and not p.name.startswith('.'))
    if not files:raise ValueError('No compact reports found.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files:archive.write(path,Path(tier)/path.relative_to(source))
    return output,len(files)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--tier',choices=('screen','paper'),required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    target,count=export(a.root,a.tier,a.output);print(f'{target}: {count} files (no state archives or figures)')

if __name__=='__main__':main()
