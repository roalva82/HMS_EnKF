#%%
import glob
import os
import zipfile
import re
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
import xarray as xr
from datetime import datetime
import matplotlib
#%%
# Paths
path_to_zip_files = './'
path_to_xml_files = './'
fields = ['Canopy Storage', 'Soil Storage']

# Get list of zip files in the target directory
zip_state_files = glob.glob(os.path.join(path_to_zip_files, 'ensemble_states_*.zip'))
zip_xml_files = glob.glob(os.path.join(path_to_xml_files, 'ensemble_simulation_*.zip'))

def read_states(zip_files, fields):
    results = []
    # Loop through zip files (skip the first one)
    for zip_path in zip_files[1:]:
        all_field_values = []  # To store all values for the fields in the current zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            try:
                with zip_ref.open('Input.state') as file:
                    text = file.read().decode('utf-8')
            except KeyError:
                print(f"Warning: 'Input.state' not found in {zip_path}")
                continue  # Skip if file doesn't exist

        #split the file in "Subbasin: ... end:" blocks    
        blocks = re.findall(r'Subbasin:[\s\S]*?End:', text)
        for block in blocks:
            subbasin = re.findall(r'Subbasin:\s*([A-Za-z0-9.]+)', block)
            for variable in fields:
                pattern = variable + r':\s*([-+]?\d*\.?\d+([eE][-+]?\d+)?)'
                values = re.findall(pattern, block)
                k = 0
                variable = variable.replace(":","") + "_"
                for value in values:
                    #fileName = str.replace(zip_ref.filename.split("\\")[-1],".zip","")
                    
                    raw_name = zip_ref.filename.split("\\")[-1]
                    fileName = raw_name.replace("_states", "").replace(".zip", "")
                    row = [fileName, subbasin[0], variable + str(k), value[0]]
                    k += 1
                    results.append(row)
   
    finalDataFrame = pd.DataFrame(results, columns=['Ensemble','Subbasin', 'Variable', 'Value'])
    finalDataFrame = pd.pivot_table(finalDataFrame, values='Value', index=['Subbasin', 'Variable'], columns='Ensemble', aggfunc="sum")
    finalDataFrame.to_csv("states.csv", header=True, index=True)  
    return finalDataFrame.reset_index()

def read_flow(zip_files):
    ns = {'pi': 'http://www.wldelft.nl/fews/PI'}

    values = []
    for zip_path in zip_files[1:]:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            try:
                with zip_ref.open('simulation.xml') as xml_file:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    for series in root.findall('.//pi:series', ns):
                        param = series.find('.//pi:parameterId', ns)
                        if param is not None and param.text == 'FLOW':
                            
                            point = series.findall('.//pi:event', ns)[-1]
                            date = point.attrib['date']
                            time = point.attrib['time']
                            value = point.attrib['value']
                            location = series.find('.//pi:locationId', ns).text
                            if date is not None and time is not None and value is not None:
                                raw_name = zip_ref.filename.split("\\")[-1]
                                fileName = raw_name.replace("_simulation", "").replace(".zip", "")
                        #        fileName = str.replace(zip_ref.filename.split("\\")[-1],".zip","")
                                format = '%Y-%m-%d_%H:%M:%S'
                                temp = datetime.strptime(date + '_' + time, format)
                                row = [location, temp, value]
                                row = [fileName, location, 'discharge', value]
                                values.append(row)

            except KeyError:
                print(f"Warning: 'simulation.xml' not found in {zip_path}")
                continue  # Skip if file doesn't exist

    finalDataFrame = pd.DataFrame(values, columns=['Ensemble','Subbasin', 'Variable', 'Value'])
    finalDataFrame = pd.pivot_table(finalDataFrame, values='Value', index=['Subbasin', 'Variable'], columns='Ensemble', aggfunc="sum")
    finalDataFrame.to_csv("simulations.csv", header=True, index=True)  
    return finalDataFrame.reset_index()

def enKF(forecast,obs_operator,observation, R):
    P = np.cov(forecast)
    #R = np.cov(observation)
    num = np.dot(P, np.transpose(obs_operator))
    den = np.dot(obs_operator, num) + R
    temp = np.linalg.inv(den)
    gain = np.dot(num, temp)
    A = forecast + np.dot(gain, (observation - np.dot(obs_operator, forecast)))
    return A

def read_observations(fileName):
    ds = xr.open_dataset(fileName)
    df = ds.FLOW.to_dataframe()
    df = df.reset_index()
    df.to_csv("FLOW.csv", header=True, index=True)  
    df.station_id = df.station_id.astype("string")
    df.station_id = df.station_id.str.split("-").str[-1].str.strip()
    df = df.sort_values(by=['time'], ascending=True)
    df = df[["station_id","FLOW"]]
    return df

#%%
statesDataFrame = read_states(zip_state_files, fields)
flowsDataFrame = read_flow(zip_xml_files)
simulations = pd.concat([statesDataFrame, flowsDataFrame], ignore_index=True)
simulations.to_csv("readStates.csv", header=True, index=False)  
observations = read_observations('dataQ.nc')
#%% example of EnKF
exclude = ['Subbasin', 'Ensemble', 'Variable']
simulation = simulations.loc[simulations['Subbasin']=='SAM01', [col for col in simulations.columns if col not in exclude]].to_numpy()
observation = observations.loc[observations['station_id']=='SAM01','FLOW'].iloc[-1]
forecast = simulation.astype(float)

ensembles = simulation.shape[1]
observation = np.repeat(observation, ensembles)
obs_operator = np.array([[0,0,1]])

obs_variance = 1e-4
#%% example of EnKF
analysis = enKF(forecast,obs_operator,observation, obs_variance)
print(forecast)
print(observation)
print(analysis)
#df.to_csv("FLOW2.csv", header=True, index=False)  
### STEP 1: READ ALL THE STATES FROM ZIP FILES - STATES AND FLOW 
#read input.state
#read simulation.xml
#build extended array
#PARA FINALES DE MAYO!


### STEP 2: READ ALL THE OBSERVATIONS
# read file dataIn.nc using xarray library
#ds = xr.open_dataset('dataIn.nc')
#Qobs = ds['Qobs'] 


### STEP 3: RUN THE ASSIMILATION USING ENKF
# run the assimilation procedure


### STEP 4: WRITE RESULTS BACK INTO TH BASIN.STATE FILE



# %%
