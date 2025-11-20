import argparse
import matplotlib.pyplot as plt
import numpy as np

def parse_log(log_content):
    lines = log_content.strip().split('\n')
    plans = []
    i = 0
    while i < len(lines):
        plan_name = lines[i].strip()
        time1 = float(lines[i + 1].strip())
        time2 = float(lines[i + 2].strip())
        average_time = (time1 + time2) / 2
        plans.append((plan_name, average_time))
        i += 3
    return plans

def plot_plans(plans, lookup_idx):
    plan, time = zip(*plans)
    plt.scatter([i for i in range(len(plan))], time, s=1)
    xtick_gap = ((len(plans) // 5)//100+1)*100
    xtick = np.arange(0, len(plans), xtick_gap)
    if abs(xtick[-1]-len(plans)) > xtick_gap:
        xtick = np.concatenate((xtick, np.array([len(plans)-1])))
    else:
        xtick[-1] = len(plans)-1
    plt.xticks(xtick)
    if lookup_idx >= 0:
        lookup_val = plans[lookup_idx][1]
        plt.axhline(lookup_val, color='r')
        error = (lookup_val-plans[0][1])/plans[0][1]
        plt.text(len(plans)/2, lookup_val, f"Error: {error*100:.1f}%", ha='center', va='bottom', color='r')
    qid = args.log.split('/')[-1][1:3]
    plt.title(f"Q{qid}")
    plt.savefig(args.log.replace('.log', '.png'))

def get_stat(plans, best_percent, min_k, max_k, lookup, shfl):
    # Sort plans by average time
    sorted_plans = sorted(plans, key=lambda x: x[1])
    lookup_idx = 0 if lookup else -1
    if lookup:
        while sorted_plans[lookup_idx][0] != lookup:
            lookup_idx += 1
        print(f"Plan {lookup} is {lookup_idx+1}/{len(sorted_plans)} "
              f"(error {(sorted_plans[lookup_idx][1]-sorted_plans[0][1])/sorted_plans[0][1]*100:.2f}%, " 
              f"top {(lookup_idx+1)/len(sorted_plans)*100:.1f}%)")
    if args.plot:
        plot_plans(sorted_plans, lookup_idx)
    if shfl:
        assert shfl.endswith('/')
        idx = 0
        while not sorted_plans[idx][0].startswith(shfl):
            idx += 1
        print(f"Best plan for {shfl} is {sorted_plans[idx][0]} and {idx+1}/{len(sorted_plans)} \
                ({(idx+1)/len(sorted_plans)*100:.1f}%)")
    best_idx = 0
    while sorted_plans[best_idx][1] < sorted_plans[0][1] * (1+best_percent/100):
        best_idx += 1
    best = sorted_plans[:best_idx]
    best_k = sorted_plans[:min_k]
    worst_k = sorted_plans[-max_k:]
    print(f"Best plans within {best_percent}% error ({len(best)}): ")
    for plan_name, avg_time in best:
        print(f"{plan_name}: {avg_time:.3f}")

    print(f"\nBest {min_k} plans: ")
    for plan_name, avg_time in best_k:
        print(f"{plan_name}: {avg_time:.3f}")

    print(f"\nWorst {max_k} plans: ")
    for plan_name, avg_time in worst_k:
        print(f"{plan_name}: {avg_time:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--best_percent', type=int, default=5)
    parser.add_argument('--min_k', type=int, default=5)
    parser.add_argument('--max_k', type=int, default=3)
    parser.add_argument('--log', required=True)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--lookup', default='')
    parser.add_argument('--shfl', default='')
    args = parser.parse_args()

    with open(args.log, "r") as log_file:
        log_content = log_file.read()

    plans = parse_log(log_content)
    get_stat(plans, args.best_percent, args.min_k, args.max_k, args.lookup, args.shfl)
