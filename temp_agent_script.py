import matplotlib.pyplot as plt
from datetime import datetime

# Data from query
data = [
    {"col_date": "2026-04-16T00:00:00", "route_count": 4},
    {"col_date": "2026-04-16T00:00:00", "route_count": 22},
    {"col_date": "2026-04-16T00:00:00", "route_count": 21},
    {"col_date": "2026-04-17T00:00:00", "route_count": 7},
    {"col_date": "2026-04-17T00:00:00", "route_count": 4},
    {"col_date": "2026-04-18T00:00:00", "route_count": 6}
]

# Aggregate by date
from collections import defaultdict
daily_totals = defaultdict(int)
for item in data:
    date_str = item["col_date"][:10]  # Extract YYYY-MM-DD
    daily_totals[date_str] += item["route_count"]

# Sort by date
sorted_dates = sorted(daily_totals.keys())
sorted_counts = [daily_totals[d] for d in sorted_dates]

# Print results
for d, c in zip(sorted_dates, sorted_counts):
    dt = datetime.strptime(d, "%Y-%m-%d")
    month_day = f"April {dt.day}"
    print(f"{month_day}: {c}")

# Calculate average
avg = sum(sorted_counts) / len(sorted_counts)
print(f"\nAverage routes per day: {avg:.1f}")

# Create line chart
plt.figure(figsize=(10, 6))
dates_labels = [f"Apr {datetime.strptime(d, '%Y-%m-%d').day}" for d in sorted_dates]
plt.plot(dates_labels, sorted_counts, marker='o', linewidth=2, markersize=8, color='#2196F3')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Routes', fontsize=12)
plt.title('Number of Routes Planned per Day - April 2026', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(0, max(sorted_counts) + 10)

# Add value labels on points
for i, v in enumerate(sorted_counts):
    plt.text(i, v + 1, str(v), ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('april_2026_routes_per_day.png', dpi=150)
print("\nChart saved as april_2026_routes_per_day.png")