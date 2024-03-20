from mesa import Agent, Model
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Model 
num_time_steps = 35040
num_houses = 10
np.random.seed(0)  # Set random seed for reproducibility

#constants for every house
T_sink = 55  # [◦C] assumed sink temperature of the heatpump
T_crawl = 12  # Crawl space temperature
f_r = 0.35  # fraction between R_exchange (T_in and T_out) and R_conduction (T_out and T_amb)
initial_T_in = 19 # Initial indoor temperature [◦C]
kWp_panel = 0.36  # kWp per panel

# Parameters houses
r_floor_range = (0.03, 0.05)        # Range for R_floor value between 0.03 and 0.05 K/W
r_exch_range = (0.002, 0.004)       # Range for R_exchange value between 0.002 and 0.004 K/W
r_cond_range = (0.003, 0.006)       # Range for R_conduction value between 0.002 and 0.004 K/W
r_vent_inf_range = (0.003, 0.007)   # Range for R_vent_inf value between 0.002 and 0.004 K/W
a_floor_range = (100, 120)          # Range for floor area between 100 and 120 m^2
volume_range = (300, 500)           # Range for volume between 300 and 500 m^3
n_panels_range = (6, 14)            # number of solar panels
run_id_range = (1,10)               # Range of run id between 1 and 10 of charging data
    
#read data and modify to read Ambient temperature
weather_data = pd.read_csv("Radiation_Temperature_2012_2022.csv", sep=";")
weather_data = weather_data["Timestamp_2012;G0_Watt_2012;Temperature_2012;Timestamp_2022;G0_Watt_2022;Temperature_2022"].str.split(";", expand=True)
weather_data.columns = ["Timestamp_2012", "G0_Watt_2012", "Temperature_2012", "Timestamp_2022", "G0_Watt_2022", "Temperature_2022"]
T_amb = pd.to_numeric(weather_data["Temperature_2012"], errors='coerce') #converts the "Temperature_2012" column to numeric format.
charging_data = pd.read_csv("charging_profiles.csv", sep=';') # Read the charging data from the CSV file
baseload_data = pd.read_csv("baseload_profiles.csv", sep=';') # Read the CSV file into a DataFrame

""" Functions 
-------------------------------------------------------------
"""

def calculate_heat_loss(initial_T_in, T_amb, f_r, r_cond,r_exch, r_floor, r_vent_inf, T_crawl, a_floor, volume, num_time_steps):
    """Calculate heatlosess and temperature of a house."""
    # Calculate C_IN and C_OUT
    C_IN = (0.055 * volume + 0.8) * 1e6  # [J/K]
    C_OUT = 3 * C_IN  # [J/K]
    
    # Calculate initial outdoor temperature
    initial_T_out = T_amb[0] + (r_cond / (r_cond + r_exch)) * (initial_T_in - T_amb[0])
    
    # Initialize lists to store temperature and heat pump values
    T_in = [initial_T_in, initial_T_in]
    T_out = [initial_T_out]
    Q_hp = [0]
    
    # Iterate over each time step starting from the second time step
    for i in range(1, num_time_steps):
        # Calculate the outdoor temperature (T_next) using the previous indoor and outdoor temperatures and other parameters
        T_onext = (1/C_OUT) * ((T_out[i-1] * (C_OUT - (1/(f_r*r_cond)) - (1/((1-f_r)*r_cond)))) + (1/(f_r*r_cond)*T_in[i-1]) + ((1/((1-f_r)*r_cond))*T_amb[i-1]))
        T_out.append(T_onext)
        
        # Calculate the difference between the current indoor temperature and the desired temperature
        error = initial_T_in - T_in[i]
        error = max(error, 0)  # Replace negative errors with 0
        
        # Calculate the heat input required to compensate for the error and keep T_in constant
        Q_hp_next = error
        Q_hp.append(Q_hp_next)
        
        # Calculate the next indoor temperature (T_next) using the current indoor and outdoor temperatures and other parameters
        T_next = ((1/C_IN) * (T_in[i] * (C_IN - (1/r_floor) - (1/(f_r*r_cond)) - (1/r_vent_inf)) + (1/r_floor) * T_crawl + (1/(f_r*r_cond)) * T_out[i] + (1/r_vent_inf) * T_amb[i])) + Q_hp[i]
        T_in.append(T_next)
        
    # Convert Q_hp to numeric format and calculate heat loss
    Q_hp = pd.to_numeric(Q_hp)
    internal_heat = 5 * a_floor  # Internal heat gain of 5W per m2 of floor area
    heatloss = Q_hp * C_IN - internal_heat  # Total heat losses [J]
    
    return heatloss, T_in, T_out

def calculate_scop(T_sink, T_amb):
    """Calculate Seasonal Coefficient of Performance (SCOP) of a heat pump."""
    delta_temp = T_sink - T_amb
    return 6.09 - 0.09 * delta_temp + 0.0005 * delta_temp**2

def calculate_power_hp(heatloss, T_sink, T_amb):
    """Calculate power consumption of the heat pump."""
    scop_series = calculate_scop(T_sink, T_amb)
    power_hp = (heatloss / scop_series) / 1000  # Convert to kW
    return power_hp

def generate_solar_profile(n_panels):
    # Define the list of solar profiles (datasets 1 to 5)
    solar_datasets = ["Solar1", "Solar2", "Solar3", "Solar4", "Solar5"]
    selected_dataset = random.choice(solar_datasets) # Randomly select one solar profile from the list

    # Read the selected solar profile from the CSV file into a DataFrame
    solar_data = pd.read_csv("ZonProfielen_2012.csv", sep=';')
    solar_data.columns = ["Timestamp_2012", *solar_datasets]  # Update column names
    selected_solar_profile = pd.to_numeric(solar_data[selected_dataset], errors='coerce')

    # Calculate solar generation based on the selected profile
    solar_generation = -selected_solar_profile * n_panels * kWp_panel
    
    return solar_generation

def get_charging_values(run_id):
    # Split the columns and convert to numeric format
    charging_data[['run_id', 'date_time', 'power', 'n']] = charging_data["run_id,date_time,power,n"].str.split(",", expand=True)
    charging_data[["power", "run_id"]] = charging_data[["power", "run_id"]].apply(pd.to_numeric, errors='coerce')
    
    # Filter the charging data for the randomly selected run_id
    selected_charging_data = charging_data[charging_data['run_id'] == run_id]
    
    # Extract the 'power' values from the filtered DataFrame
    charging_values = selected_charging_data['power'].tolist()
    
    return charging_values

def get_baseload_data(start_row_index=20, max_values=35040):
    
    baseload_list = [] # Initialize an empty list to store concatenated row data
    total_values = 0 # Initialize a variable to keep track of the total number of values
    
    for row_index in range(start_row_index, len(baseload_data)):
        # Split the row based on commas and store in separate columns
        row_data = baseload_data.iloc[row_index, 0].split(',')
        
        # Convert the row data to numeric values, excluding the first value
        row_data_numeric = [float(value) for value in row_data[1:] if value.strip()]
        
        # Filter the row data to exclude values higher than 600 or lower than 100
        filtered_row_data = [value for value in row_data_numeric if 100 <= value <= 600]
        
        # Calculate the remaining space available
        remaining_space = max_values - total_values
        
        # Check if adding all filtered values exceeds the remaining space
        if len(filtered_row_data) > remaining_space:
            # If so, truncate the filtered data to fit the remaining space
            filtered_row_data = filtered_row_data[:remaining_space]
        
        # Extend the list of concatenated row data
        baseload_list.extend(filtered_row_data)
        
        # Update the total number of values
        total_values += len(filtered_row_data)
        
        # Break the loop if the total number of values reaches max_values
        if total_values >= max_values:
            break
    
    # Convert the baseload values from W to kW
    baseload_list = [value / 1000 for value in baseload_list]
    
    return baseload_list

class House(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)                   # Call the parent class initializer
        self.unique_id = unique_id                           # Set the unique identifier for the house
        self.r_floor = random.uniform(*r_floor_range)        # Random R_floor value between specified range
        self.r_exch = random.uniform(*r_exch_range)          # Random R_exchange value between specified range
        self.r_cond = random.uniform(*r_cond_range)          # Random R_conduction value between specified range
        self.r_vent_inf = random.uniform(*r_vent_inf_range)  # Random R_vent_inf value between specified range
        self.a_floor = random.uniform(*a_floor_range)        # Random floor area between specified range
        self.volume = random.randint(*volume_range)          # Random volume between specified range
        self.n_panels = random.randint(*n_panels_range)      # number of solar panels
        self.run_id = random.randint(*run_id_range)          # random run_id of charging data between 1 and 10

    def step(self): #for every household agent
        """Calculate HP profile"""
        heat_loss, T_in, T_out = calculate_heat_loss(initial_T_in, T_amb, f_r, self.r_cond, self.r_exch, self.r_floor, self.r_vent_inf, T_crawl, self.a_floor, self.volume, num_time_steps)
        power_hp = calculate_power_hp(heat_loss, T_sink, T_amb) #calculate HP power for this time step
        self.heat_loss = heat_loss  # Assign calculated heat loss to attribute
        self.power_hp = power_hp # Assign calculated heat loss to attribute
        self.T_in = T_in
        self.T_out = T_out
        
        """generate solar profile"""
        solar = generate_solar_profile(self.n_panels)
        self.solar = solar
        
        """generate charging profile"""
        charging = get_charging_values(self.run_id)
        self.charging = charging
        
        """generate baseload profile"""
        baseload = get_baseload_data()
        self.baseload = baseload
                   
# Create an instance of the model
model = Model()
houses = [] # Create a list to store house agents

# Generate random parameters for each house
for i in range(num_houses):
    house_agent = House(unique_id=i, model=model) # Create an instance of the House agent with the generated parameters
    houses.append(house_agent)  # Append the house agent to the list
    
# Execute the step method of each House agent
for house in houses:
    house.step()
    
""" plots
--------------------------------------------------------------------
"""    

"""     
# Plot the temperature over time
plt.figure(figsize=(10, 6))
plt.plot(T_amb, label='T_ambient')
for house in houses:
    plt.plot(house.T_out, label=f'House {house.unique_id + 1} (T_out)')
    plt.plot(house.T_in, label=f'House {house.unique_id + 1} (T_in)')
plt.xlabel('Timestep')
plt.ylabel('Temperature (Degrees)')
plt.title('Temperature Variation Over Time')
plt.legend()
plt.grid(True)
plt.show()
#"""

"""
# Plot heat loss for each house
plt.figure(figsize=(10, 6))
for house in houses:
    plt.plot(house.heat_loss, label=f'House {house.unique_id + 1}')
plt.xlabel('Time Step')
plt.ylabel('Heat Loss (J)')
plt.title('Heat Loss of Houses Over Time')
plt.legend()
plt.show()
"""

"""
# Plot HP power for each house
plt.figure(figsize=(10, 6))
for house in houses:
    plt.plot(house.power_hp, label=f'House {house.unique_id + 1}')
plt.xlabel('Time Step')
plt.ylabel('Power of Heat Pump (kW)')
plt.title('Heat pump profile')
plt.legend()
plt.show()
"""

"""
# Plot solar profile for each house
plt.figure(figsize=(10, 6))
for house in houses:
    plt.plot(-house.solar, label=f'House {house.unique_id + 1}')
plt.xlabel('Time Step')
plt.ylabel('Solar generation (kW)')
plt.title('Solar profile')
plt.legend()
plt.show()
"""

"""
# Plot charging profile for each house
plt.figure(figsize=(10, 6))
for house in houses:
    plt.plot(house.charging, label=f'House {house.unique_id + 1}')
plt.xlabel('Time Step')
plt.ylabel('Charging power (kW)')
plt.title('Charging profile')
plt.legend()
plt.show()
"""

#"""
# Plot charging profile for each house
plt.figure(figsize=(10, 6))
for house in houses:
    plt.plot(house.baseload, label=f'House {house.unique_id + 1}')
plt.xlabel('Time Step')
plt.ylabel('Charging power (kW)')
plt.title('Charging profile')
plt.legend()
plt.show()
#"""