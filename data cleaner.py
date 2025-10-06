import re #whole bunch of library stuff used for data cleaning
import pandas as pd            
import numpy as np             
import joblib                
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def extract_first_number(x):    #takes the first number from a string and returns it as either a float or np.nan, numpy's representation of something that isnt a number.
    if pd.isna(x): #if nothing is found
        return np.nan #pythons version of a non applicable number
    if isinstance(x, (int, float, np.integer, np.floating)): #if x is an int float or numpy integer or float then it returns x as a float.
        return float(x)
    s = str(x) #parses x into a string.
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s) #super bizarre function that runs through a string to find the number inside. honestly still dont really understand how this works but whatever
    if m:  #if m is matched and we've found a number
        try: #literally just try this block first and then if it doesnt work try the next block, the exception block. basic exception handling in python
            return float(m.group(0)) #tries to return the string part as a float and if it fails will go straight to the exception case. 
        except:
            return np.nan
    return np.nan



def find_column(df, candidates): #helper function used to find a specified column
    cols = df.columns
    for cand in candidates:
        for c in cols:
            if c.lower() == cand.lower():  
                return c

def map_disposition_raw(v): #** change this if included another dataset
    if pd.isna(v):
        return np.nan
    s = str(v).strip().upper() #convert v into a string and remove all whitespace at the beginning then turn the entire string into uppercase letters

    if s in ('FP', 'FALSE POSITIVE', 'FALSE_POSITIVE', 'FALSEPOSITIVE') or 'FALSE' in s and 'CANDIDATE' not in s: #false positives are represented numerically by 0.
        return 0
    if s in ('PC', 'CANDIDATE', 'PLANET CANDIDATE'):
        return 1 #candidates represented by 1
    if s in ('CP', 'KP', 'CONFIRMED', 'CONFIRMED PLANET', 'CONFIRMED/KOI'):
        return 2 #confirmed planets represented by 2. May need changing if potential dataset uses any other syntax for candidate, predicted planet, and false positive.
    return np.nan
    
    
standard_columns = {  # **creating a list of keywords for each item of data to be interpreted by the ai. This must be changed if more datasets are added or removed. 
    'disposition'   : ['disposition','tfop','koi_disposition','tfopwg_disp'], #is it a false positive candidate or a confirmed planet
    'period'        : ['period','pl_orbper','koi_period'], #orbital period in days 
    'duration'      : ['duration','pl_trandurh','koi_duration'], #transit duration in hours
    'depth'         : ['depth','pl_trandep','koi_depth'], #transit depth value ppm
    'radius'        : ['radius','pl_rade','koi_prad','planet radius'], #planet radius Value [R_Earth]
    'equilibrium_t' : ['equilibrium','teq','pl_eqt', 'koi_teq'], #planet equilibrium temperature in Kelvin
    'insolation'    : ['planet insolation','pl_insol','koi_insol'], #planet insolation value [earth flux]
    'stellar_temp'  : ['stellar effective temperature','koi_steff','st_teff'], #stellar effective temperature in Kelvin
    'stellar_radius': ['st_rad','koi_srad'], #stellar radius value [R_sun]
    'stellar_logg'  : ['log(g)','st_logg','koi_slogg','stellar surface gravity'], #st log value
    'snr'           : ['snr','signal-to-noise','transit signal-to-noise','koi_model_snr','transit signal-to-noise'], #not found in tess might only be found in Kelpler
    'midpoint'      : ['midpoint','pl_tranmid','bjd','time0bk'], #transit midpoint value 
    'ra'            : ['ra','right ascension'], #Right angle in degrees and not sexagesimal
    'dec'           : ['dec','declination'], #declination in degrees
    'magnitude'     : ['st_tmag','koi_kepmag','magnitude'], #magnitude. don't know if these represent different values in the datasets
    'distance'      : ['st_dist','distance'], #no distance included in the kelpler dataset interestingly
    'impact'        : ['koi_impact'] #no impact included in the tess file interestingly
}

kepler_csv_path = "cumulative_2025.10.04_11.08.18.csv"   #kepler_csv_path is the string that represents the kepler file.
tess_csv_path   = "TOI_2025.10.05_08.35.22.csv"      #same as kepler but represents tess

dfs = [] #creating a list of dataframes
for i in (kepler_csv_path, tess_csv_path): #iterating through both CSVs with an exception handler.
    try:
        df_tmp = pd.read_csv(i, low_memory=False,comment='#') #calling pandas function to read CSVs and equalizing it to a
        print(f"Loaded {i}: {len(df_tmp)} rows")
        dfs.append(df_tmp) #appends the temporary dataframe
    except FileNotFoundError:
        print(f"File not found (skipping): {i}")

if len(dfs) == 0: #if dfs is empty we exit with the error message listed below.
    raise SystemExit("Incorrect input for the CSV. Upload file that matches code specifications")

processed_frames = []
for df in dfs: #iterates through a list of dataframes and adds them to col_map as long as the keywords match for our specified column list in standard_columns. this uses a map data structure and has columns that are found given a found value.
    col_map = {}
    for std_key, keywords in standard_columns.items():
        found = find_column(df, keywords)
        if found:
            col_map[std_key] = found
    print("Detected mapping for this file:") #debugging output verifying the column being found
    for k, v in col_map.items():
        print(f"  {k:15} -> {v}")
    new_df = pd.DataFrame() #initializes a new variable equal to pandas data frame.
    for std_key, actual_col in col_map.items():
        new_df[std_key] = df[actual_col] #renames the columns to the standardized name for ai to easily analyze without hiccups.
    if 'disposition' not in new_df.columns: #fancy way of saying if the disposition (e.g confirmed, candidate, false positive) is not found.
        cand = find_column(df, ['disposition','pdisposition','koi_disposition','koi_pdisposition','tfopwg']) #calling our helper function to check if we can find disposition with any other keywords.
        if cand: #if nothing is found this block does not run. if something is found however the block runs
            new_df['disposition'] = df[cand] #our new dataframe named disposition is appended
    processed_frames.append(new_df) #we add the new dataframe to the processed_frame list.

df_all = pd.concat(processed_frames, ignore_index=True, sort=False) #puts all the dataframes into one big old pile of dataframes
print("Combined dataframe shape:", df_all.shape) #debugging output used to tell us what shape the dataframe is combined into. shape just means how many rows and columns it contains. think of it the same way youd think of the dimensionality of a matrix. 

for col in df_all.columns: #checking columns in df_all
    if col == 'disposition':
        continue #will skip the rest of this part of the loop if the column is the disposition column. Basically don't do what is about to happen to disposition column.
    df_all[col] = df_all[col].apply(extract_first_number) #extracts the number and returns it as a float or null 

df_all['label'] = df_all['disposition'].apply(map_disposition_raw) #changes the entire disposition column into an 0s 1s and 2s based on the map disposition function from before.
print("Labels distribution (pre-drop):")
print(df_all['label'].value_counts(dropna=False)) #doesnt drop or exclude and NaN values and instead outputs them as NaNs

df_all = df_all.dropna(subset=['label']).reset_index(drop=True) #drops entire rows that dont contain the disposition column. they're garbage for the AI anyways
df_all['label'] = df_all['label'].astype(int)

missing_summary = df_all.isna().mean().sort_values(ascending=False)
print("NaN percentage by column:")
print(missing_summary.head(20)) #checking nan percentage per column for debugging purposes. 

numeric_cols = [c for c in df_all.columns if c != 'disposition' and c != 'label']
df_all[numeric_cols] = df_all[numeric_cols].apply(lambda col: col.fillna(col.median())) #this part includes the median replacement part we were discussing. If you wish to remove this then go ahead and tweak as you wish.

df_all.to_csv("cleaned_unscaled_combined.csv", index=False) #adds everything to a CSV file.
print("Saved cleaned_unscaled_combined.csv.")   #saves the file. make sure you remember this is unscaled and cleaned.

scaler = StandardScaler()
X = df_all[numeric_cols].values
X_scaled = scaler.fit_transform(X) #snippet scales everything so that the mean is 0 and the standard deviation is 1. basic scaler stuff

df_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)
df_scaled['label'] = df_all['label'].values
df_scaled.to_csv("cleaned_scaled_combined.csv", index=False)
joblib.dump(scaler, "scaler.joblib")
print("Saved cleaned_scaled_combined.csv and scaler.joblib") #this is recreating a new CSV with everything scaled and cleaned.






