"""Screen resource/time plan based on the user's measured two-chain fit time."""
import argparse, math
from research.seasonal.overnight import make_queue
from research.seasonal.job_plan import BATCHES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--minutes-per-fit',type=float,default=27.2)
    p.add_argument('--cpus',type=int,default=48)
    p.add_argument('--batch',choices=BATCHES,default='sweetspot')
    a=p.parse_args()
    if a.cpus<2 or a.minutes_per_fit<=0:p.error('Use at least two CPUs and a positive timing estimate.')
    jobs=make_queue(('screen',),a.batch)
    slots=a.cpus//2
    print(f"{len(jobs)} fits, 2 concurrent chains each; at most {slots} fits at once.")
    print(f"Equal-fit approximation: {math.ceil(len(jobs)/slots)} waves × {a.minutes_per_fit:g} minutes = "
          f"{math.ceil(len(jobs)/slots)*a.minutes_per_fit:.0f} minutes.")
    print("Allow additional time for prior simulation, forecasting and collection. Monthly fits may be slower.")
    print("This excludes scheduler waiting and CPU contention; short chains do not establish convergence.")
if __name__=='__main__':main()
