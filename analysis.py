import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv("exports/2023-11-25-sanitized-satilik.csv")
df = df[["price", "net_sqm", "created_at"]]
# Step 2: Convert the "created at" column to datetime
df["created_at"] = pd.to_datetime(df["created_at"])

# Step 4: Categorize the net_sqm into five groups
df["net_sqm_group"] = pd.cut(
    df["net_sqm"],
    bins=[0, 50, 100, 150, 200, float("inf")],
    labels=["0-50", "51-100", "101-150", "151-200", "200-above"],
)

# Step 3: Group rows by month
df["month"] = df["created_at"].dt.to_period("M")
grouped_df = df.groupby(["month", "net_sqm_group"])

# Step 5: Calculate average price for each net_sqm group within each month group
average_prices1 = grouped_df["price"].mean()

average_prices = average_prices1.unstack()

# Convert PeriodIndex to DatetimeIndex
average_prices.index = average_prices.index.to_timestamp()
# Step 6: Plot the line chart
colors = ["red", "blue", "green", "orange", "purple"]
for i, net_sqm_group in enumerate(average_prices.columns):
    breakpoint()
    plt.plot(
        average_prices.index,
        average_prices[net_sqm_group],
        color=colors[i],
        label=net_sqm_group,
    )

plt.xlabel("Month")
plt.ylabel("Average Price")
plt.legend()
plt.title("Price Change by Size Group")
plt.show()
plt.savefig("price_change_by_size_group.png")
