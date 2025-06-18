import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from scipy.interpolate import griddata

latitudes = {'STF': 72.14, 'GHB': 69.49, 'NAQ': 65.23, 'THL': 84.40,
             'SVS': 82.68, 'KUV': 80.36, 'UPN': 78.57, 'UMQ': 75.99,
             'GDH': 74.82, 'ATU': 73.54, 'SKT': 70.93, 'FHB': 66.92}

directory = "20130118-10S" #folder name that ONLY contains the 10s res data for 1/18/2013
start = pd.to_datetime('12:35:00').time()
end = pd.to_datetime('12:55:00').time()
minLat = min(latitudes.values())
maxLat = max(latitudes.values())

def stationDataProcessed(filepath, station):
    try:
        df = pd.read_csv(filepath, skiprows=17, sep=r'\s+')
        #print(df.columns)
        colH, colE = f'{station}H', f'{station}E'
        '''
        if colH not in df.columns or colE not in df.columns: #check
            return None
        '''
        #convert to datetime
        df['datetime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
        #range
        maskTime = (df['datetime'].dt.time >= start) & (df['datetime'].dt.time <= end)
        df = df[maskTime].copy()
        #checks if no data during part of choosen interval
        
        if df.empty:
            return None
        
        #20s res from 10s orginal res; change if needed
        df20s = df.iloc[::2].copy() # change to 1 for og 10 res or get rid of .iloc[::2]
        hVal = df20s[colH].values
        eVal = df20s[colE].values  
        #
        h = hVal - hVal.mean()
        e = eVal - eVal.mean()
        
        '''
        #ref time, time diff, convert
        start = df20s['datetime'].iloc[0]
        diff = df20s['datetime'] - start
        timestampArray = diff.dt.total_seconds().values
        '''
        timestampArray = (df20s['datetime'] - df20s['datetime'].iloc[0]).dt.total_seconds().values
        #print(timestampArray)
        
        H = -h  #ExB convection
        E = e   
        
        return timestampArray, latitudes[station], H, E
        
    except Exception as e:
        print(f"Error with {station}: {e}")
        return None

def stationData():
    txtData = []
    for fname in os.listdir(directory):
        if fname.endswith(".txt") and "_20130118" in fname: #txt file should be named in this form by default ex: ATU_20130118
            station = fname.split('_')[0] #gets station name
            result = stationDataProcessed(os.path.join(directory, fname), station)
            if result is not None:
                txtData.append(result)
    
    if not txtData:
        raise ValueError("station data error")
    
    #unpack list and group by varible ex: time=timestampArray.... lat=....
    times, lats, listH, listE = zip(*txtData)
    #big array where index matches 
    time = np.concatenate(times)
    #get latitude for each vector in that station to be plotted
    lat = []
    for t, l in zip(times, lats):
        lat.extend([l]*len(t))
    lat = np.array(lat)
    H = np.concatenate(listH)
    E = np.concatenate(listE)
    #print(lat)
    return txtData, time, lat, H, E

def interpolation(time, lat, H, E, timeRes):
    # Calculate scale factor
    max_magnitude = np.max(np.sqrt(H**2 + E**2))
    scale = max_magnitude * 8
    
    #time interval in seconds
    startSec = start.hour*3600 + start.minute*60 + start.second
    endSec = end.hour*3600 + end.minute*60 + end.second
    timeRangeSec = endSec - startSec #1200
    
    #calc num of pts based on the time interval and current resolution 
    num_points = int(timeRangeSec / timeRes) + 1
    #print(f"Time span: {timeRangeSec} seconds | Number of points: {num_points}")
   
    #grid
    gridTime = np.linspace(0, timeRangeSec, num_points) #61 pts from 0-1200 with 20s res
    gridLat = np.linspace(60, 90, 60) #60 lat pts from 60 to 90
    gridTimeMesh, gridLatMesh = np.meshgrid(gridTime, gridLat) #array of both
    
    #interpolate | uses nearby input pts (including 10s) to estimate valute
    gridH = griddata((time, lat), H, (gridTimeMesh, gridLatMesh), method='linear', fill_value=0)
    gridE = griddata((time, lat), E, (gridTimeMesh, gridLatMesh), method='linear', fill_value=0)
    
    #only show valid data, fill value will put 0 vectors and create little dots, this limits that
    maskLat = (gridLatMesh>=minLat) & (gridLatMesh<=maxLat)
    
    return gridTimeMesh, gridLatMesh, gridH, gridE, maskLat, scale

def plot(txtData, gridTimeMesh, gridLatMesh, gridH, gridE, maskLat, scale):
    plt.figure(figsize=(14, 8))
    #interpolated vectors
    plt.quiver(gridTimeMesh[maskLat], gridLatMesh[maskLat], gridH[maskLat], gridE[maskLat],
               width=0.001, scale=scale, headwidth=5, headlength=5, headaxislength=5,color='black')
    
    #observed from og data
    for t, lat_val, H_obs, E_obs in txtData:
        plt.quiver(t, np.full_like(t, lat_val), H_obs, E_obs,
                   width=0.001, scale=scale, headwidth=5, headlength=5,headaxislength=5, color='red')
        
    startSec = start.hour*3600 + start.minute*60 + start.second
    endSec = end.hour*3600 + end.minute*60 + end.second
    timeRangeSec = endSec - startSec #1200
    
    #change graph style if needed 
    plt.xlabel("Time (sec)")
    plt.ylabel("CGM Latitude (°)")
    plt.title("Ground Equivalent Convection Jan/18/2013 1235–1255 UT")
    plt.grid(False)
    plt.ylim(60, 90)
    plt.xlim(0, timeRangeSec)
    plt.yticks(np.arange(60, 91, 5))
    ax = plt.gca()
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.tick_params(axis='x', which='minor', length=3, color='black', width=0.5)
    ax.tick_params(axis='y', which='minor', length=3, color='black', width=0.5)
    plt.tight_layout()
    plt.show()

def main(): #have to manually change resoultion for interpolation 
    txtData, time, lat, H, E = stationData()
    gridTimeMesh, gridLatMesh, gridH, gridE, maskLat, scale = interpolation(time, lat, H, E, timeRes=20)
    plot(txtData, gridTimeMesh, gridLatMesh, gridH, gridE, maskLat, scale)

if __name__ == "__main__":
    main()