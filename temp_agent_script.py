import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Data: daily route counts for April 2026
data = {
    "2026-04-14": 20,
    "2026-04-16": 71,
    "2026-04-17": 11,
    "2026-04-18": 6
}

# Parse dates and sort
dates = sorted(data.keys())
route_counts = [data[d] for d in dates]
date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

# Calculate overall average
overall_avg = sum(route_counts) / len(route_counts)

# Print daily results
for d, c in zip(dates, route_counts):
    dt = datetime.strptime(d, "%Y-%m-%d")
    print(f"{dt.strftime('%B')} {dt.day}: {c} routes")

print(f"\nOverall average: {overall_avg:.1f} routes per calendar day")

# Create line chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(date_objs, route_counts, marker='o', linestyle='-', color='#2c7bb6', linewidth=2, markersize=8)
ax.axhline(y=overall_avg, color='red', linestyle='--', linewidth=1.5, label=f'Average: {overall_avg:.1f}')

ax.set_xlabel('Date (April 2026)')
ax.set_ylabel('Number of Routes')
ax.set_title('Routes Planned per Calendar Day - April 2026')
ax.legend()
ax.grid(True, alpha=0.3)

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.DayLocator())
fig.autofmt_xdate()

plt.tight_layout()
plt.savefig('april_2026_routes.png', dpi=150)
print("\nChart saved as april_2026_routes.png")