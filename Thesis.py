import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import scipy as sp
import math
import statistics

from scipy.signal import find_peaks
from matplotlib.widgets import Cursor

#setting parameters for plots
plt.rcParams['savefig.facecolor']='white'
plt.rcParams["figure.figsize"] = (20,7)
plt.rcParams.update({'font.size': 15})

#defining function for data import
def data_import(fibril,profile):
    distance_list = []
    profile_list = []
    #loading data from asc into 'data' array
    with open(f"/Users/vnk/Documents/{fibril}/{profile}.asc", 'rb') as f:
        data = np.loadtxt(f)

    #extracting distance values into arrays
    for i in data:
        distance_list.append(i[0])
        dist_array=np.array(distance_list)
    for j in data:
        profile_list.append(j[1])
        profile_array=np.array(profile_list)

    return (dist_array,profile_array)

#defining figure save function to avoid changing location everytime
def save_fig(name_of_figure):
    plt.savefig(f"/Users/vnk/Documents/{fibril}/{name_of_figure}.png",bbox_inches='tight',dpi=300)

#inputing location of file
region = input('Please enter region (R1/R2/R3...): ')
fibril=f'{region}/'+input(f'Please enter fibril: {region}/')
dist_raw, height_raw = data_import(fibril, 'h')
dist_raw, adh_raw = data_import(fibril, 'a')

#choosing start and end points
get_ipython().run_line_magic('matplotlib', 'qt')

#x and y arrays for definining an initial function
x = dist_raw
y = height_raw
#Plotting
raw_fig, ax1 = plt.subplots(facecolor=(1,1,1))

color = 'tab:red'
ax1.set_xlabel('Distance(nm)')
ax1.set_ylabel('Height(nm)', color=color)
ax1.plot(dist_raw, height_raw, color=color,label='Height')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yticks(np.arange(round(min(height_raw)),round(max(height_raw)+1),1));
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Adhesion(nN)', color=color)  # we already handled the x-label with ax1
ax2.plot(dist_raw,adh_raw, color=color,label='Adhesion')
ax2.tick_params(axis='y', labelcolor=color)
plt.xticks(np.arange(0,max(dist_raw),10));
ax2.grid()
# Defining the cursor
cursor = Cursor(ax2, horizOn=True, vertOn=True, useblit=True,
                color = 'r', linewidth = 1)
# Creating an annotating box
annot = ax2.annotate("", xy=(0,0), xytext=(-40,40),textcoords="offset points",
                    bbox=dict(boxstyle='round4', fc='linen',ec='k',lw=1),
                    arrowprops=dict(arrowstyle='-|>'))
annot.set_visible(False)
# Function for storing and showing the clicked values
coord = []
def onclick(event):
    global coord
    coord.append((event.xdata, event.ydata))
    x = event.xdata
    y = event.ydata
    
    # printing the values of the selected point
    print([x,y]) 
    annot.xy = (x,y)
    text = "({:.2g}, {:.2g})".format(x,y)
    annot.set_text(text)
    annot.set_visible(True)
    raw_fig.canvas.draw() #redraw the figure
    
raw_fig.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)

#slicing distance, height and adhesion arrays according to picked points
#first click = desired start point in height profile
#second click = desired end point in height profile

first_click_position = float(coord[0][0])
second_click_position = float(coord[1][0])

dist_start_point =  min(dist_raw, key=lambda x:abs(x-first_click_position))
dist_start_point_index = np.where(dist_raw == dist_start_point)
dist_end_point =  min(dist_raw, key=lambda x:abs(x-second_click_position))
dist_end_point_index = np.where(dist_raw == dist_end_point)

#saving finalized sliced distance, height and adhesion values
dist_sliced = dist_raw [dist_start_point_index[0][0]:dist_end_point_index[0][0]+1]-dist_start_point
height_sliced = height_raw [dist_start_point_index[0][0]:dist_end_point_index[0][0]+1]
adh_sliced = adh_raw [dist_start_point_index[0][0]:dist_end_point_index[0][0]+1]

get_ipython().run_line_magic('matplotlib', 'inline')
#setting parameters for plots
plt.rcParams['savefig.facecolor']='white'
plt.rcParams["figure.figsize"] = (20,7)
plt.rcParams.update({'font.size': 15})

#plotting height
plt.plot(dist_sliced,height_sliced)
plt.xticks(np.arange(min(dist_sliced),max(dist_sliced),50));
plt.xlabel('Distance (nm)')
plt.ylabel('Height (nm)')
plt.title('Raw height')
save_fig('Raw_height')

#interpolating the number of data points to 512 and saving values into dist, height and adh
dist = np.linspace(dist_sliced.min(),dist_sliced.max(),512)    
height = sp.interpolate.interp1d(dist_sliced,height_sliced,kind='cubic')(dist)
adh =  sp.interpolate.interp1d(dist_sliced,adh_sliced,kind='cubic')(dist)

#plotting height vs distance
plt.plot(dist,height)
plt.xticks(np.arange(min(dist),max(dist),50));
plt.xlabel('Distance (nm)')
plt.ylabel('Height (nm)')
plt.title('Interpolated_height')

#defining the moving_window_average function and getting the filtered D-band
def moving_window_average(x, window):
    n = len(x)
    extra = int((window-1)/2)
    x = [x[0]]*extra + x + [x[-1]]*extra
    mwa=[]
    for i in range(n):
        mwa.append(sum(x[i:i+window])/window)
    return mwa

def filtering(profile, name):
    zeroed = (profile-np.mean(profile)).tolist()
    average = moving_window_average(zeroed,3*67)
    filtered = [x1 - x2 for (x1, x2) in zip(zeroed, average)]
    
    plt.figure()
    plt.plot(dist,zeroed,label='extracted profile')
    plt.plot(dist,average,label='adjacent average')
    plt.plot(dist,filtered,label='filtered d-band')
    plt.xticks(np.arange(min(dist),max(dist),50));
    plt.xlabel('Distance (nm)')
    plt.ylabel(f'{name}')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

    return filtered

#saving filtered values of height and adhesion
height_filtered = filtering(height,'Height (nm)')

#plotting filtered d_band vs distance
plt.plot(dist,height_filtered)
plt.xticks(np.arange(min(dist),max(dist),50));
plt.xlabel('Distance(nm)')
plt.ylabel('Height(nm)')
plt.title('Filtered height')
save_fig('Filtered_height')

filtered_height_rms = np.sqrt(np.mean(np.array(height_filtered)**2))

#saving distance and corresponding filtered height values in a dictionary 'height_profile'
height_profile={}
for k in range(len(height_filtered)):
    height_profile[dist[k]]=height_filtered[k]

#saving distance values for positive and negative height values as overlap and gap resp. for FILTERED
positive_height_filtered=[]
negative_height_filtered=[]
for i in height_profile:
    if height_profile[i]>=0:
        positive_height_filtered.append(i)
    else:
        negative_height_filtered.append(i)

#FFT and saving d-band
Y=np.fft.fft(height_filtered)
freq= np.fft.fftfreq(len(height_filtered),dist[1]-dist[0])

#plotting FFT
plt.figure()
plt.plot(np.abs(freq),np.abs(Y))

#saving the value of d-band
density=list(np.abs(Y))
frequency=list(np.abs(freq))

try:
    dens=[]
    for i in frequency:
        if 0.01<=i<=0.02:
            dens.append(frequency.index(i))
    d_band=1/frequency[density.index(max([density[j] for j in dens]))]
    print(d_band)
except:
    #if the above fails, it means the d-band is not between 60-73 nm. Try this snippet below to find the value of d-band closest to 67 nm.
    print("NO D-BAND VALUE FOUND BETWEEN 60-73 nm. NOW SELECTING D-BAND AS THE CLOSEST AVAILABLE TO 67 nm")
    peaks,_ = find_peaks(density)
    frequ=[]
    for i in peaks:
        frequ.append(frequency[i])
    d=min(frequ, key=lambda x:abs(x-0.0149))
    d_band=1/d
    d_band

#sorting periods out using d_band
def period_sort(profile, name):
    periods=[]
    p={}
    period=0
    for l in profile:
        d=l-(d_band*period)
        if d<=d_band: 
            p[d]=profile[l]
        else:
            periods.append(p)
            p={}
            period+=1
            d=l-(d_band*period)
            p[d]=profile[l]
    print("Number of periods obtained="+str(period))
    
    plt.figure()
    for i in periods:
        lists=sorted(i.items())
        x,y=zip(*lists)
        plt.plot(x,y)
    plt.xlabel('Distance (nm)')
    plt.ylabel(f'{name}')
    plt.title('Line plot of '+str(period)+' periods')
    
    return period, periods

#creating height profile periods
period, height_periods = period_sort(height_profile, 'Height (nm)')
save_fig('height_periods')

#creating a list containing all values of distance
dist_arrays=[]
for b in range(len(height_periods)):
    dist_arrays.append(np.array(list(height_periods[b].keys())))
all_dist=[]
for i in dist_arrays:
    for j in i:
        all_dist.append(j)

#creating a list containing all values of height
height_arrays=[]
for c in range(len(height_periods)):
    height_arrays.append(np.array(list(height_periods[c].values())))
all_height=[]
for k in height_arrays:
    for l in k:
        all_height.append(l)

#scatter plot
plt.scatter(all_dist,all_height,s=6)
plt.xlabel('Distance(nm)')
plt.ylabel('Height(nm)')
plt.title('Scatter plot of '+str(period)+' periods')

#making a dataframe for all dist vs adh values
height_tuples = list(zip(all_dist,all_height))
height_df=pd.DataFrame(height_tuples, columns=['Distance','Height'])

#binning
#bins
def binned_average(profile_dataframe, name):
    bin_width=round((d_band/1000)*512) #bin width (window) is one d-band in pixels.
    min_value=profile_dataframe['Distance'].min()
    max_value=profile_dataframe['Distance'].max()
    bins=np.linspace(min_value,max_value,bin_width)
    a=bins[0]
    x_axis=[]
    for x in bins:
        x_axis.append((x+a)/2)
        a=x
    len(x_axis)
    #profile bins average
    y_axis=[]
    s=[]
    binn_min=[]
    binn_max=[]
    b=bins[0]
    for z in bins:
        binn=profile_dataframe[name][(float(b)<profile_dataframe['Distance']) & (profile_dataframe['Distance']<float(z))]
        s.append(np.std(binn))
        binn_min.append(np.min(binn))
        binn_max.append(np.max(binn))
        y_axis.append(np.mean(binn))
        b=z
    profile_table=pd.DataFrame(list(zip(x_axis,y_axis)),columns=['Distance',name])
    
    #removing nan values from profile_table
    profile_table=profile_table.dropna();
    
    #removing nan values from standard deviation
    std_dev = [x for x in s if math.isnan(x) == False]

    binned_average_plot, ax1 = plt.subplots(figsize=(20,7),facecolor=(1,1,1))
    ax1.plot(profile_table['Distance'],profile_table[name],'--bo',linewidth=1,markersize=4,label='d_band= '+str(round(d_band,2))+'nm')
    plt.xlabel('Distance(nm)')
    plt.ylabel(name)
    plt.title(f'{name} binned average of all periods')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
    return profile_table, std_dev, binn_min, binn_max, x_axis

#plotting binned average height
height_table, h_std_dev, h_binn_min, h_binn_max, h_x_axis = binned_average(height_df,'Height')

#plotting binned average with standard deviations, min and max
height_fig, ax2 = plt.subplots(figsize=(20,7),facecolor=(1,1,1))
ax2.plot(height_table['Distance'],height_table['Height'],'--bo',linewidth=1,markersize=4)
plt.errorbar(height_table['Distance'],height_table['Height'], h_std_dev, linestyle='None',ecolor='black',elinewidth=1,label='standard deviation')
plt.scatter(h_x_axis,h_binn_min,color='red',s=5,label='min')
plt.scatter(h_x_axis,h_binn_max,color='green',s=5,label='max')
plt.fill_between(height_table['Distance'],height_table['Height']-h_std_dev,height_table['Height']+h_std_dev,facecolor='#FFB6C1',alpha=0.4)
plt.xlabel('Distance(nm)')
plt.ylabel('Height(nm)')
plt.title('Height binned average of all periods (fingerprint)')
plt.legend(loc='upper left')
save_fig('Height_fingerprint')

fingerprint_height_range=max(height_table['Height'])-min(height_table['Height'])

mean_height_std_dev = np.mean(h_std_dev)

#saving distance values for positive and negative height values as overlap and gap resp. for FILTERED
positive_height_fingerprint=[]
negative_height_fingerprint=[]
for i in height_table['Height']:
    if i>=0:
        positive_height_fingerprint.append(float(height_table['Distance'][height_table['Height']==i]))
    else:
        negative_height_fingerprint.append(float(height_table['Distance'][height_table['Height']==i]))

# ADHESION

#plotting sliced adhesion
plt.plot(dist_sliced,adh_sliced)
plt.xticks(np.arange(min(dist_sliced),max(dist_sliced),50));
plt.xlabel('Distance (nm)')
plt.ylabel('Adhesion (nN)')
plt.title('Raw adhesion')
save_fig('Raw_adhesion')

#plotting interpolated adhesion vs distance
plt.plot(dist,adh)
plt.xticks(np.arange(min(dist),max(dist),50));
plt.xlabel('Distance (nm)')
plt.ylabel('Adhesion (nN)')
plt.title('Interpolated adhesion')

raw_mean_adh = np.mean(adh)
raw_adh_rms = np.sqrt(np.mean(np.array(adh)**2))

adh_filtered = filtering(adh,'Adhesion (nN)')

#plotting filtered d_band vs distance
plt.plot(dist,adh_filtered)
plt.xticks(np.arange(min(dist),max(dist),50));
plt.xlabel('Distance(nm)')
plt.ylabel('Adhesion(nN)')
plt.title('Adhesion filtered')
save_fig('Filtered_adhesion')

filtered_adh_rms = np.sqrt(np.mean(np.array(adh_filtered)**2))

#saving distance and corresponding adhesion(leveled) values in a dictionary 'adh_profile'
adh_profile={}
for k in range(len(adh_filtered)):
    adh_profile[dist[k]]=adh_filtered[k]

#plotting height and adhesion vs distance
combined_fig3, ax9 = plt.subplots(facecolor=(1,1,1))

color = 'tab:red'
ax9.set_xlabel('Distance(nm)')
ax9.set_ylabel('Height(nm)', color=color)
ax9.plot(height_profile.keys(),height_profile.values(), color=color,label='Height')
ax9.set_yticks(np.arange(round(min(height_profile.values())),round(max(height_profile.values())+1),1));
ax9.tick_params(axis='y', labelcolor=color)

ax10 = ax9.twinx()

color = 'tab:blue'
ax10.set_ylabel('Adhesion(nN)', color=color)
ax10.plot(adh_profile.keys(),adh_profile.values(), color=color,label='Adhesion')
ax10.tick_params(axis='y', labelcolor=color)
combined_fig3.suptitle('Filtered Height & Adhesion vs Distance')
lines = ax9.get_lines() + ax10.get_lines()
plt.text(0.98, 0.95, round(d_band),
     fontsize=18,
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax9.transAxes)
plt.xticks(np.arange(0,max(height_profile.keys()),50));
save_fig('Filtered_height_and_adhesion')

#creating adhesion periods
period, adh_periods = period_sort(adh_profile, 'Adhesion (nN)')
save_fig('adhesion_periods')

#creating a list containing all values of adhesion
adh_arrays=[]
for c in range(len(adh_periods)):
    adh_arrays.append(np.array(list(adh_periods[c].values())))
all_adh=[]
for k in adh_arrays:
    for l in k:
        all_adh.append(l)

#scatter plot of adhesion periods
plt.scatter(all_dist,all_adh,s=6)
plt.xlabel('Distance(nm)')
plt.ylabel('Adhesion(nN)')
plt.title('Scatter plot of '+str(period)+' periods')

adh_tuples = list(zip(all_dist,all_adh))
adh_df=pd.DataFrame(adh_tuples, columns=['Distance','Adhesion'])

#plotting binned average of adhesion periods
adh_table, adh_std_dev, adh_binn_min, adh_binn_max, adh_x_axis = binned_average(adh_df,'Adhesion')

#plotting binned average with standard deviations, min and max
adhesion_fig, ax4 = plt.subplots(figsize=(20,7),facecolor=(1,1,1))
ax4.plot(adh_table['Distance'],adh_table['Adhesion'],'--bo',linewidth=1,markersize=4)
plt.errorbar(adh_table['Distance'],adh_table['Adhesion'], adh_std_dev, linestyle='None',ecolor='black',elinewidth=1,label='standard deviation')
plt.scatter(adh_x_axis,adh_binn_min,color='red',s=5,label='min')
plt.scatter(adh_x_axis,adh_binn_max,color='green',s=5,label='max')
plt.fill_between(adh_table['Distance'],adh_table['Adhesion']-adh_std_dev,adh_table['Adhesion']+adh_std_dev,facecolor='#87CEFA',alpha=0.5)
plt.xlabel('Distance(nm)')
plt.ylabel('Adhesion(nN)')
plt.title('Adhesion binned average of all periods (fingerprint)')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
save_fig('Adhesion_fingerprint')

fingerprint_adh_range=max(adh_table['Adhesion'])-min(adh_table['Adhesion'])

#saving adhesion values from overlap and gap values FINGERPIRNT
overlap_adhesion_fingerprint=[]
for i in positive_height_fingerprint:
    overlap_adhesion_fingerprint.append(float(adh_table['Adhesion'][adh_table['Distance']==i]))
mean_overlap_adh = statistics.mean(overlap_adhesion_fingerprint)
mean_overlap_adh_std_dev = statistics.stdev(overlap_adhesion_fingerprint)

gap_adhesion_fingerprint=[]
for j in negative_height_fingerprint:
    gap_adhesion_fingerprint.append(float(adh_table['Adhesion'][adh_table['Distance']==j]))
mean_gap_adh = statistics.mean(gap_adhesion_fingerprint)
mean_gap_adh_std_dev = statistics.stdev(gap_adhesion_fingerprint)

#plotting binned height and adhesion together

combined_fig, ax5 = plt.subplots(facecolor=(1,1,1))

color = 'tab:red'
ax5.set_xlabel('Distance(nm)')
ax5.set_ylabel('Height(nm)', color=color)
ax5.plot(height_table['Distance'],height_table['Height'], color=color,label='Height')
ax5.tick_params(axis='y', labelcolor=color)
plt.fill_between(height_table['Distance'],height_table['Height']-h_std_dev,height_table['Height']+h_std_dev,facecolor='#FFB6C1',alpha=0.4)

ax6 = ax5.twinx()

color = 'tab:blue'
ax6.set_ylabel('Adhesion(nN)', color=color)
ax6.plot(adh_table['Distance'],adh_table['Adhesion'], color=color,label='Adhesion')
ax6.tick_params(axis='y', labelcolor=color)
plt.fill_between(adh_table['Distance'],adh_table['Adhesion']-adh_std_dev,adh_table['Adhesion']+adh_std_dev,facecolor='#87CEFA',alpha=0.5)
combined_fig.suptitle('Height & Adhesion vs Distance')
lines = ax5.get_lines() + ax6.get_lines()

combined_fig.tight_layout()  # otherwise the right y-label is slightly clipped

#stitching 3 periods
height_table_stitched=height_table
for i in range(1,3):
    height_table_propagated=pd.DataFrame(list(zip(list(height_table['Distance']+(d_band*i)),list(height_table['Height']))),columns=['Distance','Height'])
    height_table_stitched=height_table_stitched.append(height_table_propagated,ignore_index=True)
std_dev_stitched=h_std_dev*3
adh_table_stitched=adh_table
for j in range(1,3):
    adh_table_propagated=pd.DataFrame(list(zip(list(adh_table['Distance']+(d_band*j)),list(adh_table['Adhesion']))),columns=['Distance','Adhesion'])
    adh_table_stitched=adh_table_stitched.append(adh_table_propagated,ignore_index=True)
adh_std_dev_stitched=adh_std_dev*3    
#plotting binned height and adhesion together
combined_fig2, ax7 = plt.subplots(facecolor=(1,1,1))

color = 'tab:red'
ax7.set_xlabel('Distance(nm)')
ax7.set_ylabel('Height(nm)', color=color)
ax7.plot(height_table_stitched['Distance'],height_table_stitched['Height'], color=color,label='Height')
ax7.tick_params(axis='y', labelcolor=color)
ax7.set_yticks(np.arange(round(min(height_table_stitched['Height'])),round(max(height_table_stitched['Height'])+1),1));
plt.fill_between(height_table_stitched['Distance'],height_table_stitched['Height']-std_dev_stitched,height_table_stitched['Height']+std_dev_stitched,facecolor='#FFB6C1',alpha=0.4)
ax8 = ax7.twinx()

color = 'tab:blue'
ax8.set_ylabel('Adhesion(nN)', color=color)
ax8.plot(adh_table_stitched['Distance'],adh_table_stitched['Adhesion'], color=color,label='Adhesion')
ax8.tick_params(axis='y', labelcolor=color)
plt.fill_between(adh_table_stitched['Distance'],adh_table_stitched['Adhesion']-adh_std_dev_stitched,adh_table_stitched['Adhesion']+adh_std_dev_stitched,facecolor='#87CEFA',alpha=0.5)
combined_fig2.suptitle('Height & Adhesion vs Distance--3 periods stitched')
lines = ax7.get_lines() + ax8.get_lines()
plt.text(0.98, 0.95, round(d_band),
     fontsize=18,
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax7.transAxes)
plt.xticks(np.arange(0,max(height_table_stitched['Distance']),10));

#selecting the X1, X2 and X3 band
#do selection in the following order:
#1. Select the location of X1 band (the dip in the gap region).
#2. Select the point with highest adhesion in the gap region. This is done to calculate the strength of X1 band selected in the first step with respect to the point selected in this step.
#3. Select the location of X2 band (a dip in the overlap region).
#4. Select the location of X3 band (a dip in the overlap region).
get_ipython().run_line_magic('matplotlib', 'qt')
from matplotlib.widgets import Cursor
#x and y arrays for definining an initial function
x = adh_table_stitched['Distance']
y = adh_table_stitched['Adhesion']
#Plotting
combined_fig2, ax7 = plt.subplots(facecolor=(1,1,1))

color = 'tab:red'
ax7.set_xlabel('Distance(nm)')
ax7.set_ylabel('Height(nm)', color=color)
ax7.plot(height_table_stitched['Distance'],height_table_stitched['Height'], color=color,label='Height')
ax7.tick_params(axis='y', labelcolor=color)
ax7.set_yticks(np.arange(round(min(height_table_stitched['Height'])),round(max(height_table_stitched['Height'])+1),1));
#plt.fill_between(height_table_stitched['Distance'],height_table_stitched['Height']-std_dev_stitched,height_table_stitched['Height']+std_dev_stitched,facecolor='#FFB6C1',alpha=0.4)
ax8 = ax7.twinx()

color = 'tab:blue'
ax8.set_ylabel('Adhesion(nN)', color=color)
ax8.plot(adh_table_stitched['Distance'],adh_table_stitched['Adhesion'], color=color,label='Adhesion')
ax8.tick_params(axis='y', labelcolor=color)
#plt.fill_between(adh_table_stitched['Distance'],adh_table_stitched['Adhesion']-adh_std_dev_stitched,adh_table_stitched['Adhesion']+adh_std_dev_stitched,facecolor='#87CEFA',alpha=0.5)
lines = ax7.get_lines() + ax8.get_lines()
plt.xticks(np.arange(0,max(height_table_stitched['Distance']),10));
ax8.grid()
# Defining the cursor
cursor = Cursor(ax8, horizOn=True, vertOn=True, useblit=True,
                color = 'r', linewidth = 1)
# Creating an annotating box
annot = ax8.annotate("", xy=(0,0), xytext=(-40,40),textcoords="offset points",
                    bbox=dict(boxstyle='round4', fc='linen',ec='k',lw=1),
                    arrowprops=dict(arrowstyle='-|>'))
annot.set_visible(False)
# Function for storing and showing the clicked values
coord = []
def onclick(event):
    global coord
    coord.append((event.xdata, event.ydata))
    x = event.xdata
    y = event.ydata
    
    # printing the values of the selected point
    print([x,y]) 
    annot.xy = (x,y)
    text = "({:.2g}, {:.2g})".format(x,y)
    annot.set_text(text)
    annot.set_visible(True)
    combined_fig2.canvas.draw()
    
combined_fig2.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)

#position
x1pos=float(coord[0][0])
x2pos=float(coord[2][0])
x3pos=float(coord[3][0])
# xRpos=float(coord[4][0])
X1_xcoord=min(height_table_stitched['Distance'], key=lambda x:abs(x-x1pos))
X2_xcoord=min(height_table_stitched['Distance'], key=lambda x:abs(x-x2pos))
X3_xcoord=min(height_table_stitched['Distance'], key=lambda x:abs(x-x3pos))
#adhesion (have to ask upper and lower visible values for X1 calculation)
X1_adh=float(coord[0][1])-float(coord[1][1])
X2_adh=float(coord[2][1])
X3_adh=float(coord[3][1])
#height
X1_height=float(height_table_stitched['Height'][height_table_stitched['Distance']==X1_xcoord])
X2_height=float(height_table_stitched['Height'][height_table_stitched['Distance']==X2_xcoord])
X3_height=float(height_table_stitched['Height'][height_table_stitched['Distance']==X3_xcoord])
#other values
X1_lower = float(coord[0][1])
X1_upper = float(coord[1][1])
X1_strength = abs(X1_adh)
X2_strength = abs(X2_adh)
X3_strength = abs(X3_adh)

X1X2 = X1_xcoord-X2_xcoord
X2X3 = X2_xcoord-X3_xcoord
X1X3 = d_band - abs(X1_xcoord-X3_xcoord)

get_ipython().run_line_magic('matplotlib', 'inline')
#setting parameters for plots
plt.rcParams['savefig.facecolor']='white'
plt.rcParams["figure.figsize"] = (20,7)
plt.rcParams.update({'font.size': 15})

combined_fig2, ax7 = plt.subplots(facecolor=(1,1,1))

color = 'tab:red'
ax7.set_xlabel('Distance(nm)')
ax7.set_ylabel('Height(nm)', color=color)
ax7.plot(height_table_stitched['Distance'],height_table_stitched['Height'], color=color,label='Height')
ax7.tick_params(axis='y', labelcolor=color)
ax7.set_yticks(np.arange(round(min(height_table_stitched['Height'])),round(max(height_table_stitched['Height'])+1),1));
plt.fill_between(height_table_stitched['Distance'],height_table_stitched['Height']-std_dev_stitched,height_table_stitched['Height']+std_dev_stitched,facecolor='#FFB6C1',alpha=0.4)
#ax7.scatter([X1_pos,X2_pos,X3_pos],[X1_height,X2_height,X3_height],color='red')
plt.axvline(x=X1_xcoord,linestyle='--',color='green')
plt.axvline(x=X2_xcoord,linestyle='--',color='green')
plt.axvline(x=X3_xcoord,linestyle='--',color='green')
# plt.axvline(x=XR_xcoord,linestyle='--',color='green')
plt.annotate('X1', (X1_xcoord+1,max(height_table_stitched['Height'])))
plt.annotate('X2', (X2_xcoord+1,max(height_table_stitched['Height'])))
plt.annotate('X3', (X3_xcoord+1,max(height_table_stitched['Height'])))
# plt.annotate('XR', (XR_xcoord+1,max(height_table_stitched['Height'])))
ax8 = ax7.twinx()
color = 'tab:blue'
ax8.set_ylabel('Adhesion(nN)', color=color)
ax8.plot(adh_table_stitched['Distance'],adh_table_stitched['Adhesion'], color=color,label='Adhesion')
ax8.tick_params(axis='y', labelcolor=color)
plt.fill_between(adh_table_stitched['Distance'],adh_table_stitched['Adhesion']-adh_std_dev_stitched,adh_table_stitched['Adhesion']+adh_std_dev_stitched,facecolor='#87CEFA',alpha=0.5)
#ax8.scatter([X1_pos,X2_pos,X3_pos],[float(coord[0][1]),X2_adh,X3_adh],color='black')
combined_fig2.suptitle('Height & Adhesion vs Distance--3 periods stitched')
lines = ax7.get_lines() + ax8.get_lines()
#ax7.legend(lines, [line.get_label() for line in lines], loc='upper left')
plt.text(0.98, 0.95, round(d_band),
     fontsize=18,
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax7.transAxes)
plt.xticks(np.arange(0,max(height_table_stitched['Distance']),10));
save_fig('Height_adhesion_fp_stitched')