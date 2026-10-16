"""Conditional precision planning for mean specimen-level partial-stage Dice.

Scanners are repeated measurements. These calculations do not certify a final
study size without a justified variance estimate and sampling design.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import chi2, t


def required_n(sd, half_width=.05, confidence=.95):
    if not math.isfinite(sd) or not 0 < sd <= 1:
        raise ValueError('SD must be positive and <=1. Zero pilot SD cannot justify a final sample size.')
    if not 0 < half_width < 1 or not 0 < confidence < 1:
        raise ValueError('Half-width and confidence must be between zero and one.')
    def margin(n):
        return float(t.ppf((1+confidence)/2,n-1))*sd/math.sqrt(n)
    # Find the smallest integer n >= 3 using a monotone search.
    high=3
    while margin(high)>half_width:
        high*=2
        if high>10_000_000:raise ValueError('Requested precision requires more than 10 million specimens.')
    low=3
    while low<high:
        mid=(low+high)//2
        if margin(mid)<=half_width:high=mid
        else:low=mid+1
    return low,margin(low)


def pilot_specimen_scores(path, cohort='standard_groups', scanners=4):
    """Equal-weight scanner means, then independent specimen scores.

Exclude incomplete specimens rather than quietly changing scanner weights.
Both-empty plaque Dice is undefined; report exclusions explicitly.
"""
    if cohort not in ('standard_groups','G5','all') or scanners<1:
        raise ValueError('Invalid cohort or scanner count.')
    grouped=defaultdict(dict)
    with Path(path).open(encoding='utf-8-sig',newline='') as source:
        reader=csv.DictReader(source)
        needed={'physical_id','scanner','stage','plaque_dice','pilot_cohort'}
        if not needed<=set(reader.fieldnames or []):raise ValueError('Use agreement_by_scan.csv produced by the agreement analyser.')
        for row in reader:
            if row['stage']!='partial' or cohort!='all' and row['pilot_cohort']!=cohort:continue
            pid,scanner=row['physical_id'],row['scanner']
            if not pid or not scanner:raise ValueError('Missing specimen/scanner ID in partial-stage results.')
            if scanner in grouped[pid]:raise ValueError(f'Duplicate partial-stage specimen/scanner: {pid}/{scanner}')
            try:score=float(row['plaque_dice'])
            except ValueError:score=float('nan')
            if math.isfinite(score) and not 0<=score<=1:raise ValueError(f'Invalid Dice value for {pid}/{scanner}')
            grouped[pid][scanner]=score
    scores={};excluded={}
    for pid,values in sorted(grouped.items()):
        if len(values)!=scanners or not all(math.isfinite(v) for v in values.values()):
            excluded[pid]='Incomplete scanner coverage or undefined Dice; inspect the agreement report.'
        else:scores[pid]=statistics.mean(values.values())
    if len(scores)<3:raise ValueError(f'Need at least 3 complete independent pilot specimens for even a preliminary SD estimate; found {len(scores)}. Excluded: {excluded}')
    return scores,excluded


def planning_report(sd_values=None, from_results=None, plan=None, cohort=None, half_width=.05, confidence=.95, scanners=4, stages=3, available=None):
    if scanners<1 or stages<1:raise ValueError('Positive scanner/stage counts required.')
    if available is not None and available<1:raise ValueError('Available specimens must be positive.')
    if sd_values is not None and from_results is not None:raise ValueError('Choose assumed SD scenarios OR pilot results.')
    config={}
    if plan:
        config=json.loads((Path(plan)/'plan.json').read_text(encoding='utf-8'))
    cohort=cohort or config.get('precision_target',{}).get('recommended_primary_cohort','standard_groups')
    if cohort not in ('standard_groups','G5','all'):raise ValueError('Invalid cohort.')
    if plan:
        capacity=config.get('eligible_agreement_specimens_after_calibration',{})
        if available is None:available=capacity.get(cohort)
    report={'metric':'mean specimen-level partial-stage plaque Dice', 'cohort':cohort,
            'confidence':confidence,'target_half_width':half_width,'available_specimens':available,
            'status':'conditional_planning_estimate_not_precision_certification',
            'assumptions':['Independent physical specimens; scanner means weighted equally.',
                           'Student t planning approximation for specimen scores; no finite-population correction.',
                           'Dice must be defined for all specified scanners of each included specimen.',
                           'Final selection must represent the intended group/timing population.',
                           'Do not repeatedly stop when an ordinary confidence interval first becomes narrow.'],
            'scenarios':[]}
    scenarios=[]
    if from_results:
        scores,excluded=pilot_specimen_scores(from_results,cohort,scanners)
        sd=statistics.stdev(scores.values());n=len(scores)
        report['pilot']={'specimens':n,'specimen_scores':scores,'excluded_specimens':excluded,
                         'estimated_sd':sd,'source':str(Path(from_results).resolve()),
                         'sd_warning':'Pilot variance is uncertain, particularly with few specimens. No guaranteed achieved precision.'}
        if sd==0:raise ValueError('Pilot SD is zero. Diversify the pilot or supply independently justified SD scenarios; zero variation cannot certify sample size.')
        scenarios.append(('pilot_SD_point_estimate',sd))
        # A sensitivity scenario, not a guarantee for bounded/skewed Dice.
        upper=sd*math.sqrt((n-1)/float(chi2.ppf(1-confidence,n-1)))
        report['pilot']['normal_model_one_sided_SD_upper_bound']=upper
        report['pilot']['normal_model_warning']='The chi-square SD bound assumes normally distributed independent specimen scores; small/skewed Dice pilots may violate this.'
        scenarios.append(('normal_model_SD_upper_sensitivity',max(sd,min(upper,.5))))
        if upper>.5:report['pilot']['SD_upper_capped_at_population_bound']=.5
    else:
        scenarios=[('assumed_SD_not_observed',float(sd)) for sd in (sd_values if sd_values is not None else (.05,.10,.15))]
    for basis,sd in scenarios:
        n,margin=required_n(sd,half_width,confidence)
        report['scenarios'].append({'basis':basis,'sd':sd,'required_physical_specimens':n,
                                   'all_stage_scans_per_rater':n*scanners*stages,
                                   'primary_partial_scans_per_rater':n*scanners,
                                   'conditional_half_width_at_n':margin,
                                   'within_available_pool':None if available is None else n<=available})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    variance=p.add_mutually_exclusive_group()
    variance.add_argument('--sd',type=float,nargs='+',help='Hypothetical or externally justified specimen SDs.')
    variance.add_argument('--from-results',type=Path,help='agreement_by_scan.csv from an independently rated pilot.')
    p.add_argument('--plan',type=Path,help='Study plan directory; reads eligible specimen capacities.')
    p.add_argument('--cohort',choices=('standard_groups','G5','all'),help='Default: cohort specified in the plan, otherwise standard_groups.')
    p.add_argument('--half-width',type=float,default=.05);p.add_argument('--confidence',type=float,default=.95)
    p.add_argument('--scanners',type=int,default=4);p.add_argument('--stages',type=int,default=3)
    p.add_argument('--available',type=int);p.add_argument('--output',type=Path,help='Optional NEW JSON report path; never overwrites.')
    args=p.parse_args()
    try:
        report=planning_report(args.sd,args.from_results,args.plan,args.cohort,args.half_width,args.confidence,args.scanners,args.stages,args.available)
        if args.output:
            with args.output.open('x',encoding='utf-8') as target:json.dump(report,target,indent=2)
        print(json.dumps(report,indent=2));return 0
    except Exception as exc:print('ERROR:',exc,file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
