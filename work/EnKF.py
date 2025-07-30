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

path_to_zip_files = './'#sys.argv[1]
path_to_xml_files = './'#sys.argv[2]
fields = ['Canopy Storage', 'Surface Storage', 'Soil Storage', 'Groundwater Storage']

idmap = {
    'SAM01':'FLORID',
    'SCA01':'CARAPA',
    'SFF01':'SAOFFA',
    'SFV01':'SAOFVE',
    'SIG01':'ESTIGU',
    'SIH01':'IVINHE',
    'SIH02':'IVINHE',
    'SIH03':'IVINHE',
    'SIH04':'IVINHE',
    'SIV01':'TERCRI',
    'SIV02':'UBASUL',
    'SIV03':'BARFER',
    'SIV04':'PORPNO',
    'SIV05':'NOVPTA',
    'SPQ01':'PEQMUN',
    'SPQ02':'BALCAN',
    'SPQ03':'NOVPDO',
    'SPQ04':'NOVBSM'
}

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
    return A, P, gain

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

def update_block(match):
    block = match.group(0)
    subbasin_id = match.group(2)

    values = storage_values.get(subbasin_id)

    if not values:
        return block  # leave unchanged if no values

    # Update Groundwater Storage values
    if "Groundwater Storage" in values:
        gw_values = values["Groundwater Storage"]
        
        if gw_values is None or (isinstance(gw_values, float) and np.isnan(gw_values)):
            return block
        else:
            # Replacement function that consumes values in order
            def replace_gw(m):
                if replace_gw.index < len(gw_values):
                    val = gw_values[replace_gw.index]
                    replace_gw.index += 1
                    return m.group(1) + str(val)
                else:
                    return m.group(0)  # no change if we run out of values

            replace_gw.index = 0
            block = re.sub(
                r"(Groundwater Storage:\s*)([^\s]+)",
                replace_gw,
                block
            )

    # Update all other fields that have a unique value
    else:
        for field in fields:
            if field in values:
                value = values[field]
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    return block
                else:
                    block = re.sub(
                        r"({field}:\s*)([^\s]+)",
                        lambda m: m.group(1) + str(values[field]),
                        block
                    )

    return block

#%%
statesDataFrame = read_states(zip_state_files, fields)
flowsDataFrame = read_flow(zip_xml_files)
simulations = pd.concat([statesDataFrame, flowsDataFrame], ignore_index=True)
try:
    simulations.to_csv("readStates.csv", header=True, index=False)
except:
    pass

observations = read_observations('dataQ.nc')

exclude = ['Subbasin', 'Variable']
storage_values = {}

for sub, flow in zip(idmap.keys(), idmap.values()):
    simulation = simulations.loc[simulations['Subbasin']==sub, [col for col in simulations.columns if col not in exclude]].to_numpy().astype(float)
    observation = observations.loc[observations['station_id']==flow,'FLOW'].iloc[-1]
    ensembles = simulation.shape[1]
    observation = np.repeat(observation, ensembles)

    obs_operator = np.zeros((1,len(simulation)))
    obs_operator[0,-1] = 1

    obs_variance = 1e-4 * np.mean(simulation, axis=1)[-1]
    analysis, covariance, gain = enKF(simulation,obs_operator,observation, obs_variance)
    
    mean_analysis = np.mean(analysis, axis=1)

    print('Subbasin: ', sub, '\n')
    print('obs operator\n', obs_operator)
    print('covariance\n', covariance)
    print('gain\n', gain)
    print('simulation\n', simulation)
    print('obs\n', observation)
    print('analysis\n', analysis)
    print('mean analysis', mean_analysis, '\n \n')

    values = [mean_analysis[0], mean_analysis[1], mean_analysis[2], [mean_analysis[3], mean_analysis[4]]]   # depends on declared fields!!!
    storage_values[sub] = dict(zip(fields, values))


zip_path = 'ensemble_states_0.zip'
file_to_extract = 'Input.state'
extract_to = './updates/'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    if file_to_extract in zip_ref.namelist():
        zip_ref.extract(file_to_extract, extract_to)

with open(extract_to + file_to_extract, 'r') as file:
    content = file.read()
    pattern = r"(Subbasin:\s*(\w+)(?:.|\n)*?End:)"
    updated_text = re.sub(pattern, update_block, content)


with open(extract_to + file_to_extract, 'w') as file:
    file.write(updated_text)



'''
test = simulations.loc[simulations['Subbasin']=='SAM01', [col for col in simulations.columns if col in exclude]].to_numpy()
df = pd.DataFrame([mean_analysis], columns=test)
print(df)


####COMENTARIO PARA SERGIO: CONVERTIR mean_analysis A UN DICCIONARIO COMO EL SIGUIENTE:
storage_values = {
    "SIH02": {
        "Soil Storage": 1.1,
        "Groundwater Storage": [0.01, 0.02]
    },
    "SIH03": {
        "Soil Storage": 2.2,
        "Groundwater Storage": [0.03, 0.04]
    }
}

'''

'''
## WRITE STATES BACK TO THE TEXT
# Pattern to extract each Subbasin block
"


# Apply replacements

print(updated_text)
'''
