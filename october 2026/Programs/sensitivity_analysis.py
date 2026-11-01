import sys
import random
import collections

# Update this path if needed
sys.path.insert(0, r"C:\Users\fmanw\Desktop\old laptop\traffic\Real Model\Programs")
import image_processing_pipeline as P

# Define a realistic "Congested" scenario
road_info = {
    "name": "Sensitivity Test Road",
    "length": 100, "width": 10, "lanes": 2,
    "current_cars": 24, "capacity": 30, "avg_speed": 40,
    "congestion_times": "8-9 AM", "special_events": [],
    "has_crosswalk": True, "has_bikelane": True, "condition": "Fair",
    "restrictions": ["no new lanes"]
}

# 1. Baseline: NO detections
base_top_solutions = P.analyze_traffic(road_info, [])[2]
base_top1 = base_top_solutions[0]

print(f"Base Top-1 (No detections): {base_top1}\n")

# 2. Monte Carlo: Simulate noisy CNN detections
n_trials = 500
top1_changed = 0
new_top1_counts = collections.Counter()

for _ in range(n_trials):
    # Simulate noisy CNN: 50% chance a class is detected, score between 0.35 and 0.9
    detected = []
    for c in P.CLASSES:
        if random.random() > 0.5:
            detected.append((c, random.uniform(0.35, 0.90)))
            
    _, _, top_solutions = P.analyze_traffic(road_info, detected)
    if top_solutions[0] != base_top1:
        top1_changed += 1
        new_top1_counts[top_solutions[0]] += 1

print(f"Top-1 Recommendation changed in {top1_changed / n_trials * 100:.1f}% of noisy trials.")
print("\nMost common replacements for Top-1:")
for sol, count in new_top1_counts.most_common(5):
    print(f"  - {sol}: {count / n_trials * 100:.1f}%")