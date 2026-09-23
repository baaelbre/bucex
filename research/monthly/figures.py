"""Regenerate the manuscript figures from report CSVs, without fitting."""
import argparse
from pathlib import Path
import bucex as bx
from .report import new_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True,
                        help='One joint report, one independent run root, or several series directories.')
    parser.add_argument('--config', type=Path, default=Path('research/monthly/config/figures.json'))
    parser.add_argument('--output', type=Path, default=Path('results/serra_figures'))
    parser.add_argument('--series', nargs='+', choices=tuple(bx.UCCLE_INFO))
    parser.add_argument('--panels', nargs='+', help='Select figure names from the recipe configuration.')
    parser.add_argument('--formats', nargs='+', choices=('png','pdf','svg'))
    parser.add_argument('--strict', action='store_true', help='Fail on missing exports instead of drawing placeholders.')
    args = parser.parse_args()
    config = bx.load_config(args.config)
    reports = bx.ReportCollection.from_directories(args.reports, series=args.series or config['series_order'])
    recipes = config['recipes']
    if args.panels:
        unknown = set(args.panels)-{recipe['name'] for recipe in recipes}
        if unknown:
            parser.error(f'Unknown figure names: {sorted(unknown)}')
        recipes = [recipe for recipe in recipes if recipe['name'] in args.panels]
    directory = new_run(args.output, 'manuscript')
    bx.save_config(config, directory/'figure_config.json')
    result = bx.save_publication_figures(reports, directory, recipes=recipes,
        colors=config.get('colors'), formats=args.formats or config.get('formats',['png']),
        dpi=config.get('dpi',180), style=config.get('style','manuscript'), strict=args.strict)
    for item in result['figures']:
        print(f"{item['name']}: {item['status']}")
    print(directory)


if __name__ == '__main__':
    main()
