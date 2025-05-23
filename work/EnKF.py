import glob
import os
import zipfile
import re
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
import xarray as xr

# Paths
path_to_zip_files = './'
path_to_xml_files = './'
fields = ['Canopy Storage', 'Soil Storage']

# Get list of zip files in the target directory
zip_state_files = glob.glob(os.path.join(path_to_zip_files, 'ensemble_states_*.zip'))
zip_xml_files = glob.glob(os.path.join(path_to_xml_files, 'ensemble_simulation_*.zip'))

def extract_state(text, keyword):
    """Extract numeric values associated with a given keyword."""
    pattern = fr"{re.escape(keyword)}:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    return [float(val) for val in re.findall(pattern, text)]

def extract_state_name(text, keyword): #NO FUNCTIONA TODAVIA
    """Extract string names associated with a given keyword."""
    pattern = r'Subbasin:\s*([A-Za-z0-9.]+)'
    return [string(el) for el in re.findall(pattern, text)]


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
                pattern = variable + r'\s*([-+]?\d*\.?\d+([eE][-+]?\d+)?)'
                values = re.findall(pattern, block)
                k = 0
                variable = variable.replace(":","") + "_"
                for value in values:
                    fileName = str.replace(zip_ref.filename.split("\\")[-1],".zip","")
                    row = [fileName, subbasin[0], variable + str(k), value[0]]
                    k += 1
                    results.append(row)
   
    finalDataFrame = pd.DataFrame(results, columns=['Ensemble','Subbasin', 'Variable', 'Value'])
    finalDataFrame.to_csv("variables1.csv", header=True, index=True)  
    finalDataFrame = pd.pivot_table(finalDataFrame, values='Value', index=['Subbasin', 'Variable'], columns='Ensemble', aggfunc="sum")
    finalDataFrame.to_csv("variables2.csv", header=True, index=True)  
    return finalDataFrame

def read_states_OLD(zip_files, fields):
    # Prepare a list to store all results for each zip file
    results = []

    # Loop through zip files (skip the first one)
    for zip_path in zip_files[1:]:
        all_field_values = []  # To store all values for the fields in the current zip file
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            try:
                # Ensure the file exists before attempting to read
                with zip_ref.open('Input.state') as file:
                    text = file.read().decode('utf-8')
            except KeyError:
                print(f"Warning: 'Input.state' not found in {zip_path}")
                continue  # Skip if file doesn't exist

        # For each field, extract its values and store them in a list
        for field in fields:
            state_values = extract_state(text, field)
            all_field_values.extend(state_values)  # Add all values for this field to the list

        # Append the list of field values (flattened) for the current zip file to results
        results.append(all_field_values)

    # Convert the final results list into a NumPy array
    final_array = np.array(results, dtype=object)

    # Output the resulting array
    return(final_array)

def read_flow(zip_files):
    ns = {'pi': 'http://www.wldelft.nl/fews/PI'}

    # Initialize dictionary to store time series per file
    data_dict = {}

    for zip_path in zip_files:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            try:
                with zip_ref.open('simulation.xml') as xml_file:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()

                    for series in root.findall('.//pi:series', ns):
                        param = series.find('.//pi:parameterId', ns)
                        if param is not None and param.text == 'FLOW':
                            point = series.findall('.//pi:event', ns)[-1] # takes the last value
                            date = point.attrib['date']
                            time = point.attrib['time']
                            value = point.attrib['value']

                            label = os.path.basename(zip_path)
                            data_dict[label] = pd.Series(data=value)
                            pd.Series(data=value)


for series in root.findall('.//pi:series', ns):
    param = series.find('.//pi:parameterId', ns)
    if param is not None and param.text == 'FLOW':
        values = []
        for point in series.findall('.//pi:event', ns):
            date = point.attrib['date']
            time = point.attrib['time']
            value = point.attrib['value']
            if date is not None and time is not None and value is not None:
                format = '%Y-%m-%d_%H:%M:%S'
                temp = datetime.strptime(date + '_' + time, format)
                values.append((temp, value))
        flow_series.append(values)


            except KeyError:
                print(f"Warning: 'simulation.xml' not found in {zip_path}")
                continue  # Skip if file doesn't exist

    # Combine all series into a DataFrame, aligned by timestamps
    df = pd.DataFrame(data_dict)
    return(df)

def enKF(forecast,obs_operator,observation):
    P = np.cov(forecast)
    R = np.cov(observation)
    num = np.dot(P,np.transpose(obs_operator))
    den = np.dot(obs_operator,num) + R
    temp = np.linalg.inv(den)
    gain = np.dot(num,temp)
    A = forecast + np.dot(gain,(observation-np.dot(obs_operator,forecast)))
    return A

#read_states(zip_state_files, fields)

read_flow(zip_xml_files)


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


