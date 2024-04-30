import pandas as pd

# Load the log file
log_df = pd.read_csv('gpu_power_usage.log', names=['timestamp', 'power_usage_watts'])

# Convert power usage to kilowatts
log_df['power_usage_kW'] = log_df['power_usage_watts'] / 1000

# Calculate the total energy consumption in kWh
# Assuming that the log is in seconds intervals
total_energy_kWh = log_df['power_usage_kW'].sum() / (60 * 60)  # Divide by 3600 to convert seconds to hours

print(f"Total energy consumption: {total_energy_kWh} kWh")
