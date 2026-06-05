import matplotlib.pyplot as plt
from datetime import datetime

data = [
    {"col_date": "2026-04-17T00:00:00", "route_count": 7},
    {"col_date": "2026-04-18T00:00:00", "route_count": 6},
    {"col_date": "2026-04-17T00:00:00", "route_count": 4},
    {"col_date": "2026-04-16T00:00:00", "route_count": 4},
    {"col_date": "2026-04-16T00:00:00", "route_count": 22},
    {"col_date": "2026-04-16T00:00:00", "route_count": 21}
]

# Aggregate by date
from collections import defaultdict
daily_totals = defaultdict(int)
for d in data:
    date_str = d["col_date"][:10]  # Extract YYYY-MM-DD
    daily_totals[date_str] += d["route_count"]

# Sort by date
sorted_dates = sorted(daily_totals.keys())
sorted_counts = [daily_totals[d] for d in sorted_dates]

# Calculate average
avg_routes = sum(sorted_counts) / len(sorted_counts)

print("Daily route counts:")
for d, c in zip(sorted_dates, sorted_counts):
    # Format as "April 16: 47" etc.
    dt = datetime.strptime(d, "%Y-%m-%d")
    month_day = dt.strftime("%B %-d")
    print(f"{month_day}: {c}")

print(f"\nAverage number of routes per day: {avg_routes:.2f}")

# Create line chart
plt.figure(figsize=(10, 6))
dates_formatted = [datetime.strptime(d, "%Y-%m-%d").strftime("%b %-d") for d in sorted_dates]
plt.plot(dates_formatted, sorted_counts, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
plt.xlabel('Date')
plt.ylabel('Number of Routes')
plt.title('Number of Routes Planned per Day in April 2026')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('april_2026_routes_per_day.png')
print("\nChart saved as april_2026_routes_per_day.png")