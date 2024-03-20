import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


#read data and modify to read temperature and solar profile
weather_data = pd.read_csv("Radiation_Temperature_2012_2022.csv", sep=";")
weather_data = weather_data["Timestamp_2012;G0_Watt_2012;Temperature_2012;Timestamp_2022;G0_Watt_2022;Temperature_2022"].str.split(";", expand=True)
weather_data.columns = ["Timestamp_2012", "G0_Watt_2012", "Temperature_2012", "Timestamp_2022", "G0_Watt_2022", "Temperature_2022"]
T_amb = pd.to_numeric(weather_data["Temperature_2012"], errors='coerce') #converts the "Temperature_2012" column to numeric format.

# Timesteps
num_time_steps = 35040

#constants for every house
T_sink = 55  # [◦C] assumed sink temperature of the heatpump
T_crawl = 12  # Crawl space temperature
f_r = 0.35  # fraction between R_exchange (T_in and T_out) and R_conduction (T_out and T_amb)
initial_T_in = 19 # Initial indoor temperature [◦C]

# Used parameters
r_floor = 0.04 # R_floor [K/W]
r_exch = 0.003 # R_exchange [K/W]
r_cond = 0.004 # R_conduction [K/W]
r_vent_inf = 0.005 # R_ventilation+infiltration [K/W]
a_floor = 100  # Floor area in m2
volume = 400

C_IN = (0.055 * volume + 0.8) * 1e6  # [J/K]
C_OUT = 3 * C_IN  # [J/K]

initial_T_out = T_amb[0] + (r_cond / (r_cond + r_exch)) * (initial_T_in - T_amb[0]) # Calculate outside temperature initial 
internal_heat = 5 * a_floor  # internal heat gain of 5W per m2 of floor area  
    
# Initialize an array to store temperature values for each time step
T_in = [initial_T_in,initial_T_in]  # Assuming initial_T_in is the initial temperature
T_out = [initial_T_out] # Assuming initial_T_out is the initial temperature
Q_hp = [0] # Assuming no heat input in the initial step

for i in range(1, num_time_steps): # Iterate over each time step starting from the second time step

    #Calculate the outdoor temperature (T_next) using the previous indoor and outdoor temperatures and other parameters
    T_onext = (1/C_OUT) * ((T_out[i-1] * (C_OUT - (1/(f_r*r_cond)) - (1/((1-f_r)*r_cond)))) + (1/(f_r*r_cond)*T_in[i-1]) + ((1/((1-f_r)*r_cond))*T_amb[i-1]))

    T_out.append(T_onext) # Append the calculated temperature for the next time step to the list  
    
    # Calculate the difference between the current indoor temperature and the desired temperature
    error = initial_T_in - T_in[i] 
    error = max(error, 0) # Replace negative errors with 0
    
    # Calculate the heat input required to compensate for the error and keep T_in constant
    Q_hp_next = error # Adjust the heat input based on the error
    Q_hp.append(Q_hp_next)  # Append the calculated heat input for the next time step to the list
    
    #Calculate the next indoor temperature (T_next) using the current indoor and outdoor temperatures and other parameters
    T_next = ((1/C_IN) * (T_in[i] * (C_IN - (1/r_floor) - (1/(f_r*r_cond)) - (1/r_vent_inf)) + (1/r_floor) * T_crawl + (1/(f_r*r_cond)) * T_out[i] + (1/r_vent_inf) * T_amb[i])) + Q_hp[i]
            
    T_in.append(T_next) # Append the calculated temperature for the next time step to the list


Q_hp2 = pd.to_numeric(Q_hp) #converts the column to numeric format.
heatloss = Q_hp2*C_IN - internal_heat #total heatlosses [J]

def calculate_scop(T_sink,T_amb): #calculate seasonal COP Heatpump

    delta_temp = T_sink - T_amb
    return pd.Series(6.09 - 0.09 * delta_temp + 0.0005 * delta_temp**2) #copsh = 6.09 − 0.09 · ∆Tsink−amb + 0.0005 · ∆Tsink−amb

scop_series = calculate_scop(T_sink, T_amb) #calculate COP Heatpump 

Power_HP = (heatloss/scop_series)/1000 #power HP in kW
HP_total = sum(Power_HP)/4
#print("HP =", HP_total)


""" reading baseload data
--------------------------------------------------------------------
"""
baseload_data = pd.read_csv("baseload_profiles.csv", sep=';') # Read the CSV file into a DataFrame
start_row_index = 20  # Start reading data from row 21
baseload_list = [] # Initialize an empty list to store concatenated row data
total_values = 0 # Initialize a variable to keep track of the total number of values

for row_index in range(start_row_index, len(baseload_data)): # Iterate over the rows in the DataFrame
    # Split the row based on commas and store in separate columns
    row_data = baseload_data.iloc[row_index, 0].split(',')
    
    # Convert the row data to numeric values, excluding the first value
    row_data_numeric = [float(value) for value in row_data[1:] if value.strip()]
    
    # Filter the row data to exclude values higher than 600 or lower than 100
    filtered_row_data = [value for value in row_data_numeric if 100 <= value <= 600]
    
    # Calculate the remaining space available
    remaining_space = 35040 - total_values
    
    # Check if adding all filtered values exceeds the remaining space
    if len(filtered_row_data) > remaining_space:
        # If so, truncate the filtered data to fit the remaining space
        filtered_row_data = filtered_row_data[:remaining_space]
    
    # Extend the list of concatenated row data
    baseload_list.extend(filtered_row_data)
    
    # Update the total number of values
    total_values += len(filtered_row_data)
    
    # Break the loop if the total number of values reaches 35040
    if total_values >= 35040:
        break
    
baseload_list = [value / 1000 for value in baseload_list] #kW
baseload_total = sum(baseload_list)/4
#print("Baseload =",baseload_total)

""" charging profiles
--------------------------------------------------------------------
"""    
    
charging_data = pd.read_csv("charging_profile.csv", sep=';') # Read the CSV file into a DataFrame

# Split the columns "run_id", "date_time", "power", and "n" by comma and expand into separate columns
charging_data[['run_id', 'date_time', 'power', 'n']] = charging_data["run_id,date_time,power,n"].str.split(",", expand=True)
charging_data["power"] = pd.to_numeric(charging_data["power"], errors='coerce') # Convert the "power" column to numeric format
charging_data = charging_data.head(35040) # Include only the first 35,040 values
charge_total = charging_data["power"].sum()/4
#print("Charging =",charge_total)

""" solar profiles
--------------------------------------------------------------------
"""    

n_panels = 8 #number of solar panels
kWp_panel = 0.36 #wp

solar_data = pd.read_csv("ZonProfielen_2012.csv", sep=';') # Read the CSV file into a DataFrame
solar_data.columns = ["Timestamp_2012", "Solar1", "Solar2", "Solar3", "Solar4", "Solar5"]
Solar = pd.to_numeric(solar_data["Solar1"], errors='coerce')
Solar = -Solar * n_panels * kWp_panel
Solar_total = sum(Solar)/4
#print("Solar =",Solar_total)

""" plots
--------------------------------------------------------------------
"""    

#"""     
# Plot the temperature over time
plt.figure(figsize=(10, 6))
plt.plot(T_amb, label='T_ambient')
plt.plot(T_out, label='T_outside')
plt.plot(T_in, label='T_inside')
plt.xlabel('Timestep')
plt.ylabel('Temperature (Degrees)')
plt.title('Temperature Variation Over Time')
plt.legend()
plt.grid(True)
plt.show()
#"""

"""     
# Plot the heat_loss over time
plt.figure(figsize=(10, 6))
plt.plot(heatloss, label='heatloss')
plt.xlabel('Timestep')
plt.ylabel('Heatloss (Joules)')
plt.title('Heatloss Over Time')
plt.legend()
plt.grid(True)
plt.show()
"""

"""
# Plot the power of the heat pump for each timestep
plt.figure(figsize=(10, 6))
plt.plot(Power_HP)
plt.xlabel('Timestep')
plt.ylabel('Power of Heat Pump (kW)')
plt.title('Heat pump profile')
plt.grid(True)
plt.show()
"""

"""  
# Power_HP and T_amb for each timestep
fig, ax1 = plt.subplots(figsize=(10, 6))
# Plot Power_HP on the primary y-axis
ax1.plot(Power_HP, color='blue', label='Power of Heat Pump (kW)')
ax1.set_xlabel('Timestep')
ax1.set_ylabel('Power of Heat Pump (kW)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
# Create a secondary y-axis for T_amb
ax2 = ax1.twinx()
ax2.plot(T_amb, color='grey', label='Ambient Temperature')
ax2.plot(T_in, color='black', label='Inside Temperature')
ax2.set_ylabel('Ambient Temperature', color='grey')
ax2.tick_params(axis='y', labelcolor='black')
# Show legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title('Heat Pump Profile and Temperature')
plt.grid(True)
plt.show()  
"""    

"""
# Plot the solar_profiles over time
plt.figure(figsize=(10, 6))
plt.plot(Solar, label='Solar generation profile')  # Assign a label to the plot
plt.xlabel('Timestep')
plt.ylabel('Solae generation (kW)')
plt.title('Solar generation profile')
plt.legend()  # Include the legend based on the labels assigned
plt.grid(True)
plt.show()
"""

"""
# Plot the baseload data
plt.figure(figsize=(10, 6))
plt.plot(baseload_list)
plt.xlabel('Timestep')
plt.ylabel('Load (kW)')
plt.title('Baseload profile')
plt.grid(True)
plt.show()
"""

"""
# Plot the charging data
plt.figure(figsize=(10, 6))
plt.plot(charging_data['power'])
plt.xlabel('Timestep')
plt.ylabel('Load (kW)')
plt.title('Charging profile')
plt.grid(True)
plt.show()
"""

"""
# Calculate the total power at each timestep
total_power = np.array(baseload_list) + np.array(Power_HP) + np.array(charging_data['power']) + np.array(Solar)
# Plot the total power for each timestep
plt.figure(figsize=(10, 6))
plt.plot(range(len(total_power)), total_power, label='Total Power')
plt.xlabel('Timestep')
plt.ylabel('Power (kW)')
plt.title('Combined Profiles')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
"""

