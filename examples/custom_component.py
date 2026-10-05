"""An external linear state component, without changes to fitting or prediction."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import bucex as bx


@bx.register_type
@dataclass(frozen=True)
class DampedLevel:
    persistence: float = .9
    name: str = 'damped'

    def build(self, steps, priors, exog=None):
        if not 0 < self.persistence < 1:
            raise ValueError('persistence must lie between zero and one.')
        return bx.ComponentBlock(
            name=self.name,
            transition=np.array([[self.persistence]]),
            design=np.ones((steps,1)),
            initial=np.eye(1),
            coefficient_names=(self.name+'.initial',),
            priors=(bx.Normal(0,2),),
            state_names=(self.name+'.state',),
            innovations=(bx.Innovation(self.name,np.eye(1),bx.Normal(0,.2)),),
        )


def main():
    model=bx.Model(bx.Gaussian(),[DampedLevel()],bx.Priors(variance=bx.Fixed(.2**2)))
    y=bx.simulate(model,30,seed=8).y['y'][0]
    fit=bx.fit(y,model=model,mcmc=bx.MCMC(draws=30,warmup=20,chains=2,progress=False))
    output=Path('results/examples/custom');output.mkdir(parents=True,exist_ok=True)
    restored=bx.load(fit.save(output/'fit.bucex'))
    bx.plot(restored,type='component',component='damped',path=output/'damped.png')
    print(restored.predict(10,draws=50).path('damped').shape)


if __name__=='__main__': main()
