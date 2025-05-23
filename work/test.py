import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd

# Path to your XML file
xml_file = 'simulation.xml'



# Parse the XML
tree = ET.parse(xml_file)
root = tree.getroot()

# Namespace handling
ns = {'pi': 'http://www.wldelft.nl/fews/PI'}

# Find all time series entries with parameterId == FLOW
flow_series = pd.array(['date','time','value'])

for series in root.findall('.//pi:series', ns):
    param = series.find('.//pi:parameterId', ns)
    if param is not None and param.text == 'FLOW':
        values = []
        for point in series.findall('.//pi:event', ns):
            date = point.attrib['date']
            time = point.attrib['time']
            value = point.attrib['value']
            location = series.find('.//pi:locationId', ns).text
            if date is not None and time is not None and value is not None:
                format = '%Y-%m-%d_%H:%M:%S'
                temp = datetime.strptime(date + '_' + time, format)
                values.append((temp, value, location))
        flow_series['value'] = 

print(flow_series[0])
