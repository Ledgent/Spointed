import osmnx as ox
import pandas as pd
from geopy.geocoders import Nominatim
import numpy as np

# Step 1: Load your dataset
file_path = 'static/data/locations.csv'  # Path to your CSV file
df = pd.read_csv(file_path)

# Step 2: Simulate new features

# Add population density (people per sq km)
np.random.seed(42)  # Ensure reproducibility
df['population_density'] = np.random.randint(10000, 50000, size=len(df))

# Add income level (average monthly income category)
df['income_level'] = np.random.choice(['low', 'medium', 'high'], size=len(df), p=[0.4, 0.4, 0.2])

# Add proximity to points of interest (simulated categorical data)
df['proximity_to_poi'] = np.random.choice(['near_offices', 'near_malls', 'near_residential'], size=len(df))

# Add safety index (scale of 1-10)
df['safety_index'] = np.random.randint(1, 11, size=len(df))

# Add parking availability (Yes/No)
df['parking_availability'] = np.random.choice(['Yes', 'No'], size=len(df), p=[0.7, 0.3])

# Step 3: Integrate real-time data

# Initialize geolocator
geolocator = Nominatim(user_agent="restaurant_locator")

# Function to get coordinates from state or location
def get_coordinates(location):
    try:
        loc = geolocator.geocode(location)
        return loc.latitude, loc.longitude
    except:
        return None, None

# Add latitude and longitude to the dataset
df['coordinates'] = df['state'].apply(get_coordinates)

# Separate latitude and longitude into two columns
df['latitude'] = df['coordinates'].apply(lambda x: x[0] if x is not None else None)
df['longitude'] = df['coordinates'].apply(lambda x: x[1] if x is not None else None)

# Drop the combined 'coordinates' column
df = df.drop(columns=['coordinates'])

# Fetch amenities using OSMnx
def fetch_nearby_amenities(lat, lon, amenity_type='restaurant', distance=500):
    try:
        # Get a graph around the specified latitude and longitude
        point = (lat, lon)
        amenities = ox.geometries_from_point(point, tags={amenity_type: True}, dist=distance)
        return len(amenities)  # Count of nearby amenities
    except:
        return 0

# Add a column for the count of nearby restaurants within 500 meters
df['nearby_restaurants'] = df.apply(
    lambda row: fetch_nearby_amenities(row['latitude'], row['longitude'], amenity_type='restaurant', distance=500)
    if pd.notnull(row['latitude']) and pd.notnull(row['longitude']) else 0, axis=1)

# Step 4: Save the updated dataset
output_file = 'static/data/locations_with_features.csv'  # Name for the updated file
df.to_csv(output_file, index=False)

print(f"Updated dataset with features and real-time data saved as {output_file}")
