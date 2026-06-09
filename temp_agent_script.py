import json
from datetime import datetime
from collections import defaultdict

# Raw data from query
data = [
    {"col_date": "2026-04-17T00:00:00", "route_count": 7},
    {"col_date": "2026-04-18T00:00:00", "route_count": 6},
    {"col_date": "2026-04-17T00:00:00", "route_count": 4},
    {"col_date": "2026-04-16T00:00:00", "route_count": 4},
    {"col_date": "2026-04-16T00:00:00", "route_count": 22},
    {"col_date": "2026-04-16T00:00:00", "route_count": 21},
]

# Aggregate route counts by date
daily_totals = defaultdict(int)
for row in data:
    date_str = row["col_date"][:10]  # Extract YYYY-MM-DD
    daily_totals[date_str] += row["route_count"]

# Sort by date
sorted_dates = sorted(daily_totals.keys())
sorted_counts = [daily_totals[d] for d in sorted_dates]

# Print results in the requested format
print("Daily Route Counts for April 2026:")
for d in sorted_dates:
    dt = datetime.strptime(d, "%Y-%m-%d")
    print(f"April {dt.day}: {daily_totals[d]} routes")

# Calculate overall average
total_routes = sum(sorted_counts)
num_days = len(sorted_dates)
avg = total_routes / num_days
print(f"\nOverall average: {avg:.1f} routes per calendar day")

# Generate line chart
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

x_labels = [f"April {datetime.strptime(d, '%Y-%m-%d').day}" for d in sorted_dates]
ax.plot(x_labels, sorted_counts, marker='o', linestyle='-', color='#2c7bb6', linewidth=2, markersize=8)

# Add average line
ax.axhline(y=avg, color='#d7191c', linestyle='--', linewidth=1.5, label=f'Average: {avg:.1f} routes/day')

ax.set_xlabel('Calendar Day', fontsize=12)
ax.set_ylabel('Number of Routes', fontsize=12)
ax.set_title('Routes Planned per Calendar Day - April 2026', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Annotate each point
for i, count in enumerate(sorted_counts):
    ax.annotate(str(count), (x_labels[i], count), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('april_2026_routes.png', dpi=150)
print("\nChart saved as april_2026_routes.png")