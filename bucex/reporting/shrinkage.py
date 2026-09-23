"""Compact, chain-preserving reports of learned common regularization."""
from pathlib import Path

from ..diagnostics.shrinkage import compare_shared_shrinkage, compare_initial_slope_priors
from ..diagnostics.sensitivity import innovation_prior_diagnostics
from ..priors.shrinkage import NORMAL_ABSOLUTE_MEDIAN
from ..plotting import plot_chain_traces, trace_frame, publication_style


def save_shared_shrinkage_report(fit, directory, *, level=.95, figures=True,
                                 style="manuscript", dpi=180, seed=182,
                                 horizon=360, rate_multiplier=120.,
                                 response_unit="response units", rate_unit=None):
    """Write hyperprior/posterior, updating, trace and diagnostic exports."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    comparison = compare_shared_shrinkage(fit, level=level)
    comparison.to_csv(directory/"shared_shrinkage.csv", index=False)
    import pandas as pd
    period = fit.model.period
    calibration = pd.DataFrame(fit.priors.shrinkage.calibration(period=period,
        horizon=horizon, slope_time_unit=rate_multiplier, unit=response_unit))
    calibration.to_csv(
        directory/"shared_shrinkage_calibration.csv", index=False)
    effects = comparison.copy()
    for c in effects.component.unique():
        row = calibration[calibration.component.eq(c)].iloc[0]
        factor = rate_multiplier if c == "initial_slope" else row.response_gain / NORMAL_ABSOLUTE_MEDIAN
        mask = effects.component.eq(c)
        effects.loc[mask, ['lower','median','upper','anchor']] *= factor
        effects.loc[mask, 'scale'] = 'initial_rate_prior_SD' if c == 'initial_slope' else 'future_contribution_prior_SD'
    effects.assign(horizon_updates=horizon, rate_multiplier=rate_multiplier).to_csv(
        directory/'shared_shrinkage_effects.csv',index=False)
    initial = compare_initial_slope_priors(fit, seed=seed, rate_multiplier=rate_multiplier, level=level)
    initial.to_csv(directory/'initial_slope_prior_posterior.csv',index=False)
    innovation_prior_diagnostics(comparison).to_csv(directory/"shared_shrinkage_updates.csv", index=False)
    draws = {key: value for key, value in fit.parameter_draws.items() if key.startswith("shrinkage.shared.")}
    trace_frame(draws).to_csv(directory/"shared_shrinkage_traces.csv.gz", index=False)
    if figures:
        import matplotlib.pyplot as plt
        with publication_style(style=style, dpi=dpi):
            figure, _ = plot_chain_traces(draws)
            figure.savefig(directory/"shared_shrinkage_traces.png", dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            if not initial.empty:
                figure, ax = plt.subplots(figsize=(7, 3.8), layout='constrained')
                names = list(initial.channel.drop_duplicates())
                for i, name in enumerate(names):
                    for distribution, offset, color in [('prior',-.14,'.6'),('posterior',.14,'C0')]:
                        row = initial[(initial.channel==name)&(initial.distribution==distribution)].iloc[0]
                        ax.errorbar(row['median'],i+offset,
                            xerr=[[row['median']-row['lower']],[row['upper']-row['median']]],
                            fmt='o',color=color,label=distribution if i==0 else None)
                ax.axvline(0,color='.5',lw=.8)
                ax.set(yticks=range(len(names)),yticklabels=names,
                    xlabel='initial slope / '+(rate_unit or f'{response_unit} per {rate_multiplier:g} updates'))
                ax.legend()
                figure.savefig(directory/'initial_slope_prior_posterior.png',dpi=dpi,bbox_inches='tight')
                plt.close(figure)
            components = comparison.component.drop_duplicates()
            figure, axes = plt.subplots(1, len(components), squeeze=False,
                figsize=(5*len(components), 3.2), layout="constrained")
            for ax, component in zip(axes[0], components):
                for i, row in enumerate(comparison[comparison.component.eq(component)].to_dict("records")):
                    ax.errorbar(row["median"], i,
                        xerr=[[row["median"]-row["lower"]], [row["upper"]-row["median"]]], fmt="o")
                ax.set(yticks=[0,1], yticklabels=["hyperprior", "posterior"],
                    xlabel="initial slope prior SD" if component == "initial_slope" else f"shared {component} SD median")
                ax.ticklabel_format(axis="x", style="sci", scilimits=(-3,3), useMathText=True)
            figure.savefig(directory/"shared_shrinkage.png", dpi=dpi, bbox_inches="tight")
            plt.close(figure)
    return directory


__all__ = ["save_shared_shrinkage_report"]
