import pandas as pd
import numpy as np
import matplotlib as plt
# from fuzzywuzzy import fuzz


#Dataset

#-------------------------------------------AviationDataset------------------------------------------

aviationData = pd.read_csv("/Users/nishanth_p/Desktop/Thesis/Code/Data/AviationData.csv", encoding='ISO-8859-1')

#aviationData.dropna(subset=['Total.Fatal.Injuries', 'Total.Serious.Injuries' ,'Total.Minor.Injuries' , 'Total.Uninjured'], inplace=True)

aviationData['Aboard'] = (aviationData['Total.Fatal.Injuries'] +
                              aviationData['Total.Serious.Injuries'] +
                              aviationData['Total.Minor.Injuries'] +
                              aviationData['Total.Uninjured'])

aviationData['Fatalities'] = aviationData['Total.Fatal.Injuries']


aviationData['Event.Date'] = pd.to_datetime(aviationData['Event.Date'], errors='coerce')
#--------Broad Phase of Flight-----
import pandas as pd

#Get unique values from Broad.phase.of.flight

valid_phases = set(aviationData['Broad.phase.of.flight'].dropna().unique())

# Function to check if any word in Report.Status matches a valid phase
def fill_with_report_status(row):
    if pd.isna(row['Broad.phase.of.flight']):
        # Check if Report.Status is a string before splitting
        if isinstance(row['Report.Status'], str) and row['Report.Status']:
            # Split Report.Status into words and check if any of the words is in valid_phases
            status_words = set(row['Report.Status'].split())
            matched_phases = valid_phases.intersection(status_words)
            
            if matched_phases:
                return matched_phases.pop()  # Take one of the matched phases (in case there are multiple)
    return row['Broad.phase.of.flight']

# Apply the function to update missing values
aviationData['Broad.phase.of.flight'] = aviationData.apply(fill_with_report_status, axis=1)



#--------------Drop columns-----------

columns_to_drop = ['Total.Fatal.Injuries', 
                   'Total.Serious.Injuries', 
                   'Total.Minor.Injuries', 
                   'Total.Uninjured', 
                   'Event.Id', 
                   'Accident.Number', 
                   'Airport.Code', 
                   'Airport.Name', 
                   'FAR.Description', 
                   'Schedule', 
                   'Publication.Date',
                   'Report.Status',
                   "Air.carrier"]

# Drop the columns from the dataset
aviationData.drop(columns_to_drop, axis=1, inplace=True)

#---------Rename columns to ensure they match in both datasets
aviationData = aviationData.rename(columns={
    'Event.Date': 'Date',
    'Location': 'Location',
    'Country': 'Country',
    'Make': 'Make',
    'Model': 'Model',
    'Aboard': 'Aboard',
    'Fatalities': 'Fatalities'
})



#---------------------------------------------------------AirplaneCrashesAndFatalities----------------------------------------

airplaneCrashes = pd.read_csv("/Users/nishanth_p/Desktop/Thesis/Code/Data/Airplane_Crashes_and_Fatalities_Since_1908.csv", encoding='ISO-8859-1')

airplaneCrashes['Date'] = pd.to_datetime(airplaneCrashes['Date'], errors='coerce')

#--------Location and Country--------
#Split the "Location" column at the last comma
airplaneCrashes[['Location', 'Country']] = airplaneCrashes['Location'].str.rsplit(',', n=1, expand=True)

# Strip any leading or trailing whitespace
airplaneCrashes['Country'] = airplaneCrashes['Country'].str.strip()

#-------------MakeAndModel-----

# Define a custom split function with NaN check
def custom_split(value):
    if isinstance(value, str):  # Check if the value is a string
        if 'De Havilland' in value:
            # Split after "De Havilland"
            parts = value.split('De Havilland', 1)
            return ['De Havilland' + parts[1].split(' ', 1)[0], parts[1].split(' ', 1)[1] if len(parts[1].split(' ', 1)) > 1 else '']
        else:
            # Split at the first space for other entries
            return value.split(' ', 1)
    else:
        # If the value is not a string (NaN or float), return empty strings
        return ['', '']

# Apply the custom split function
airplaneCrashes[['Make', 'Model']] = airplaneCrashes['Type'].apply(custom_split).apply(pd.Series)

# Clean up any extra spaces
airplaneCrashes['Make'] = airplaneCrashes['Make'].str.strip()
airplaneCrashes['Model'] = airplaneCrashes['Model'].str.strip()

airplaneCrashes.drop("Type", axis=1, inplace=True)

#----DropColumns----

airplaneCrashes.drop(columns= ['Time', 'Operator','Flight #','Route','Registration','cn/In','Ground','Summary'],inplace=True)

#---------Rename columns to ensure they match in both datasets
airplaneCrashes = airplaneCrashes.rename(columns={
    'Event.Date': 'Date',
    'Location': 'Location',
    'Country': 'Country',
    'Make': 'Make',
    'Model': 'Model',
    'Aboard': 'Aboard',
    'Fatalities': 'Fatalities'
})


#----------------------------------------------MergeDatasets---------------------------------------------

# Merge the datasets on the matching columns
df = pd.merge(airplaneCrashes, aviationData,
                       on=['Date', 'Location', 'Country', 'Make', 'Model', 'Aboard', 'Fatalities'],
                       how='outer')  # or 'outer', 'left', 'right' depending on your needs


df['SurvivalRate'] = (df['Aboard'] - df['Fatalities']) / df['Aboard']
df.to_csv("yearGraph.csv", index=False)

df.dropna(subset=['Latitude','Longitude','Registration.Number','SurvivalRate'], inplace=True)

df.drop(['Broad.phase.of.flight', "Injury.Severity", "Aboard", "Fatalities"],axis=1,inplace=True)

#-----fill with random sampling-------

# Function to fill NA values in a column based on existing value distribution
def fill_na_with_distribution(df, column_name):
    # Get the value distribution using value_counts
    value_counts = df[column_name].value_counts(normalize=True)
    # Fill NA values based on this distribution
    df[column_name] = df[column_name].apply(
        lambda x: np.random.choice(value_counts.index, p=value_counts.values) if pd.isna(x) else x
    )

# List of columns to fill
columns_to_fill = ['Engine.Type', 'Number.of.Engines', 'Aircraft.damage', 'Aircraft.Category', 'Purpose.of.flight','Weather.Condition']

# Apply the function to each column in the list
for column in columns_to_fill:
    fill_na_with_distribution(df, column)

#----fill with mode------
# Function to fill NA values with the most frequent value in a column
def fill_na_with_mode(df, column_name):
    # Get the most frequent value (mode) in the column
    mode_value = df[column_name].mode()[0]
    # Fill NA values with this mode value
    df[column_name].fillna(mode_value, inplace=True)

# List of columns to fill
columns_to_fill = ['Amateur.Built', 'Make', 'Model']

# Apply the function to each column in the list
for column in columns_to_fill:
    fill_na_with_mode(df, column)

df.to_csv("MergedDataset.csv", index=False)

#------------------------------------------------Updated Dataset------------------------------------------------------------------

df = pd.read_csv("MergedDataset.csv")

# df['Date'] = pd.to_datetime(df['Date'])
# df['Year'] = df['Date'].dt.year
# df['Month'] = df['Date'].dt.month
# df['Weekday'] = df['Date'].dt.weekday 
# df.drop(['Date'],axis=1 ,inplace=True)

df.rename(columns={
    'Location': 'Location',
    'Country': 'Country',
    'Make': 'Aircraft_Make',
    'Model': 'Aircraft_Model',
    'Investigation.Type': 'Investigation_Type',
    'Latitude': 'Latitude',
    'Longitude': 'Longitude',
    'Aircraft.damage': 'Aircraft_Damage',
    'Aircraft.Category': 'Aircraft_Category',
    'Registration.Number': 'Registration_Number',
    'Amateur.Built': 'Amateur_Built',
    'Number.of.Engines': 'Number_of_Engines',
    'Engine.Type': 'Engine_Type',
    'Purpose.of.flight': 'Purpose_of_Flight',
    'Weather.Condition': 'Weather_Condition',
    'SurvivalRate': 'Survival_Rate',
    'Year': 'Year',
    'Month': 'Month',
    'Weekday': 'Day'  # You can keep 'Weekday' if you prefer
}, inplace=True)

#df.drop(['Location','Country'],axis=1, inplace=True)

df.drop(['Location'],axis=1, inplace=True)

# Function to convert DMS (Degrees Minutes Seconds) format to decimal degrees
def dms_to_decimal(dms_coord):
    # If the value is missing (NaN) or not a string, return NaN
    # if pd.isna(dms_coord) or not isinstance(dms_coord, str):
    #     return np.nan
    
    if '.' in dms_coord:
        try:
            return float(dms_coord)  # Convert the string to float if it's already in decimal
        except ValueError:
            return np.nan

    # Check for valid DMS format (must end in N, S, E, or W)
    direction = dms_coord[-1]
    if direction not in ['N', 'S', 'E', 'W']:
        return np.nan

    # Extract degrees, minutes, seconds from the coordinate string
    dms = dms_coord[:-1]
    if len(dms) < 6:
        return np.nan  # Ensure there are enough characters for DMS format

    try:
        degrees = int(dms[:2])
        minutes = int(dms[2:4])
        seconds = int(dms[4:])
    except ValueError:
        return np.nan  # Return NaN if parsing fails

    # Calculate decimal degrees
    decimal = degrees + (minutes / 60) + (seconds / 3600)
    
    # Adjust for direction (negative for 'S' or 'W')
    if direction in ['S', 'W']:
        decimal = -decimal

    return decimal


# Apply the conversion function to your existing DataFrame's Latitude and Longitude columns
df['Latitude'] = df['Latitude'].apply(dms_to_decimal)
df['Longitude'] = df['Longitude'].apply(dms_to_decimal)


# Define a mapping for Purpose_of_Flight categories
purpose_mapping = {
    'Personal': 'Personal and Instructional',
    'Instructional': 'Personal and Instructional',
    'Aerial Application': 'Commercial and Business',
    'Business': 'Commercial and Business',
    'Positioning': 'Commercial and Business',
    'Other Work Use': 'Commercial and Business',
    'Aerial Observation': 'Commercial and Business',
    'Flight Test': 'Commercial and Business',
    'Executive/corporate': 'Commercial and Business',
    'Ferry': 'Commercial and Business',
    'Skydiving': 'Commercial and Business',
    'External Load': 'Commercial and Business',
    'Air Race show': 'Commercial and Business',
    'Banner Tow': 'Commercial and Business',
    'Public Aircraft - Federal': 'Government/Public Service',
    'Public Aircraft - Local': 'Government/Public Service',
    'Public Aircraft - State': 'Government/Public Service',
    'Public Aircraft': 'Government/Public Service',
    'Glider Tow': 'Special Operations',
    'Firefighting': 'Special Operations',
    'Air Drop': 'Special Operations',
    'ASHO': 'Special Operations',
    'PUBS': 'Special Operations',
    'PUBL': 'Special Operations',
    'Unknown': 'Other'  # Handling the unknown cases
}

# Apply the mapping to create a new column
df['Purpose_of_Flight'] = df['Purpose_of_Flight'].map(purpose_mapping)

category_mapping = {
    'Airplane': 'Fixed-wing Aircraft',
    'Glider': 'Fixed-wing Aircraft',
    'Helicopter': 'Rotary-wing Aircraft',
    'Gyrocraft': 'Rotary-wing Aircraft',
    'Balloon': 'Light Aircraft',
    'Weight-Shift': 'Light Aircraft',
    'Powered Parachute': 'Light Aircraft',
    'Ultralight': 'Light Aircraft',
    'WSFT': 'Light Aircraft',
    'Rocket': 'Experimental and Special Aircraft',
    'Blimp': 'Experimental and Special Aircraft',
    'ULTR': 'Experimental and Special Aircraft',
    'Unknown': 'Unknown'
}

# Apply the mapping to create a new column
df['Aircraft_Category'] = df['Aircraft_Category'].map(category_mapping)


def categorize_engine_type(engine_type):
    if engine_type == 'Geared Turbofan' or engine_type == 'Turbo Fan' or engine_type == "Turbofan":
        return 'Turbo Fan'
    elif engine_type in ['Unknown', 'Electric', 'LR', 'NONE', 'Hybrid Rocket', 'UNK']:
        return 'Others'
    else:
        return engine_type  # Keep original type if it doesn't fall in the specified groups

# Apply categorization function to Engine.Type column
df['Engine_Type'] = df['Engine_Type'].apply(categorize_engine_type)

df['Aircraft_Make'] = df['Aircraft_Make'].str.lower()


# Step 1: Standardize capitalization
df['Aircraft_Make'] = df['Aircraft_Make'].str.lower()

# Step 2: Create a function to group similar categories
def group_similar_categories(categories, threshold=90):
    groups = []
    checked = set()  # To avoid re-checking
    for cat in categories:
        if cat not in checked:
            group = [cat]  # Initialize a new group
            checked.add(cat)
            for other_cat in categories:
                if other_cat != cat and fuzz.ratio(cat, other_cat) >= threshold:  # Compare similarity
                    group.append(other_cat)
                    checked.add(other_cat)
            groups.append(group)
    return groups

# Get the unique categories
categories = df['Aircraft_Make'].unique()

# Group similar categories
similar_groups = group_similar_categories(categories)

# Create a mapping of original categories to their new grouped names
category_mapping = {}
for group in similar_groups:
    group_name = group[0]  # Use the first category as the group name
    for category in group:
        category_mapping[category] = group_name

# Step 3: Replace the original categories with their grouped name
df['Aircraft_Make'] = df['Aircraft_Make'].map(category_mapping)

df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.weekday 
df.drop(['Date'],axis=1 ,inplace=True)

df.to_csv("Final_data.csv", index=False)





