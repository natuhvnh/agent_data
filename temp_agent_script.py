import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Data
dates = [datetime(2026, 4, 17), datetime(2026, 4, 18)]
route_counts = [11, 6]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(dates, route_counts, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)

# Formatting
plt.title('Number of Routes Planned per Day - April 2026', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Routes', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Format x-axis dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.xticks(rotation=45)

# Set y-axis to integer ticks
plt.yticks(range(0, max(route_counts) + 2, 1))

# Add data labels
for i, (d, c) in enumerate(zip(dates, route_counts)):
    plt.text(d, c + 0.3, str(c), ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('april_2026_routes_per_day.png', dpi=150)
plt.show()