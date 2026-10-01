"""JSON-controlled sensitivity plan, independent jobs and comparison figures."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import bucex as bx
from .configuration import CONFIG, read_config, merge, validate
from .run import run


def plan(path=CONFIG/'sensitivity.json'):
    path = Path(path)
    spec = json.loads(path.read_text())
    if set(spec) != {'base', 'variants'}:
        raise ValueError('Sensitivity JSON must contain only base and variants.')
    base = validate(read_config(path.parent/spec['base']))
    names = [v['name'] for v in spec['variants']]
    if len(set(names)) != len(names) or any('/' in n or '\\' in n for n in names):
        raise ValueError('Sensitivity variant names must be unique directory-safe names.')
    configs = []
    for v in spec['variants']:
        if set(v) != {'name', 'override'}:
            raise ValueError('A sensitivity variant requires name and override.')
        c = validate(merge(base, v['override']))
        c['name'] = v['name']
        configs.append(c)
    return configs


def table(configs):
    return pd.DataFrame([dict(index=i, name=c['name'], pooling=c['pooling'],
        **c['priors'], seasonal=c['model']['seasonal'], scale_prior_sd=c['model']['scale_prior_sd'])
        for i, c in enumerate(configs)])


def compare(root, configs):
    from .figures import style, panels, savefig
    import matplotlib.pyplot as plt
    style()
    root = Path(root)
    fits = {c['name']: bx.load(root/c['name']/'fit.bucex') for c in configs if (root/c['name']/'fit.bucex').exists()}
    if not fits:
        raise ValueError('No completed sensitivity fits found.')
    first = next(iter(fits.values()))
    names, rows = list(first.channels), []
    for kind in ('level', 'slope', 'seasonal'):
        fig, axes = panels(names)
        for name, ax in zip(names, axes):
            curves = []
            for i, (variant, f) in enumerate(fits.items()):
                if not np.array_equal(f[name].index, first[name].index):
                    raise ValueError('Sensitivity fits must use the same observations and calendar.')
                color = plt.get_cmap('viridis')(i/max(1, len(fits)-1))
                # Native plot includes each setting's uncertainty; light bands
                # distinguish it from a claim of identical full posteriors.
                bx.plot(f, channel=name, type=kind, ax=ax, color=color, label=variant)
                curves.append(f[name].path(kind).mean(axis=(0,1)))
            rows.append(dict(channel=name, component=kind, fits=len(fits),
                max_mean_path_spread=float(np.ptp(np.stack(curves), axis=0).max())))
        fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=3, fontsize=6)
        savefig(fig, root/'figures', 'sensitivity_'+kind, configs[0])
    pd.DataFrame(rows).to_csv(root/'path_comparison.csv', index=False)
    table(configs).assign(completed=lambda x: x['name'].isin(fits)).to_csv(root/'sensitivity_settings.csv', index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=CONFIG/'sensitivity.json')
    p.add_argument('--profile', choices=('smoke','screen','paper'), default='screen')
    p.add_argument('--output', type=Path, default=Path('results/sensitivity'))
    p.add_argument('--variant', help='Variant name, or zero-based index for a scheduler array.')
    p.add_argument('--workers', type=int)
    p.add_argument('--list', action='store_true')
    p.add_argument('--compare', action='store_true')
    args = p.parse_args()
    configs = plan(args.config)
    if args.list:
        print(table(configs).to_string(index=False)); return
    root = args.output/args.profile
    if args.compare:
        compare(root, configs); return
    selected = configs
    if args.variant is not None:
        selected = [configs[int(args.variant)]] if args.variant.isdigit() else [c for c in configs if c['name'] == args.variant]
        if not selected:
            raise ValueError('Unknown sensitivity variant.')
    root.mkdir(parents=True, exist_ok=True)
    table(configs).to_csv(root/'sensitivity_settings.csv', index=False)
    for config in selected:
        run(config, profile=args.profile, output=root/config['name'], workers=args.workers, figures=False)


if __name__ == '__main__':
    main()
