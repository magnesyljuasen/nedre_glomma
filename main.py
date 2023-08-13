import pandas as pd
import numpy as np
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient
import swifter
import ast

class Constants:
    def __init__(self):
        self.BUILDING_TYPES = {
            "A": "Hou",
            "B": "Apt",
            "C": "Off",
            "D": "Shp",
            "E": "Htl",
            "F": "Kdg",
            "G": "Sch",
            "H": "Uni",
            "I": "CuS",
            "J": "Nsh",
            "K": "Other",
            "L": "Other",
        }

        self.DEKNINGSGRADER_GSHP = {
            'A' : 100, 
            'B' : 90,
            "C" : 90,
            "D" : 90,
            "E" : 90,
            "F" : 90,
            "G" : 90,
            "H" : 90,
            "I" : 90,
            "J" : 90,
            "K" : 90,
            "L" : 90,
            }

        self.COEFFICIENT_OF_PERFORMANCES_GSHP = {
            'A' : 3.5, 
            'B' : 3.5,
            "C" : 3.5,
            "D" : 3.5,
            "E" : 3.5,
            "F" : 3.5,
            "G" : 3.5,
            "H" : 3.5,
            "I" : 3.5,
            "J" : 3.5,
            "K" : 3.5,
            "L" : 3.5,
            }
        
        self.COEFFICIENT_OF_PERFORMANCES_ASHP = {
            'A' : 2.2, 
            'B' : 2.2,
            "C" : 2.2,
            "D" : 2.2,
            "E" : 2.2,
            "F" : 2.2,
            "G" : 2.2,
            "H" : 2.2,
            "I" : 2.2,
            "J" : 2.2,
            "K" : 2.2,
            "L" : 2.2,
            }

        self.BUILDING_STANDARDS = {
            "X": "Reg", 
            "Y": "Eff-E", 
            "Z": "Vef"}

        self.SOLARPANEL_DATA = pd.read_csv('src/solenergi_antakelser.csv', sep = ";")
        self.SOLARPANEL_BUILDINGS = {
            'A' : 'Småhus', 
            'B' : 'Boligblokk',
            'C' : 'Næringsbygg_mindre',
            'D' : 'Næringsbygg_større',
            'E' : 'Småhus', 
            'F' : 'Boligblokk',
            'G' : 'Næringsbygg_mindre',
            'I' : 'Næringsbygg_større',
            'J' : 'Småhus', 
            'K' : 'Boligblokk',
            'L' : 'Næringsbygg_mindre',
        }

        self.PROFET_DATA = pd.read_csv('src/profet_data.csv', sep = ";")

        self.GSHP = 'grunnvarme'
        self.SOLAR_PANELS = 'solceller'
        self.AIR_SOURCE_HEAT_PUMP = 'luft_luft_varmepumpe'
        self.DISTRICT_HEATING = 'fjernvarme'
        self.BUILDING_STANDARD = 'byggstandard_profet'
        self.BUILDING_TYPE = 'byggtype_profet'
        self.AREA = 'areal'
        self.TEMPERATURE_ARRAY = '_utetemperatur'
        self.THERMAL_DEMAND = '_termisk_energibehov'
        self.ELECTRIC_DEMAND = '_elspesifikt_energibehov'
        self.COMPRESSOR = '_kompressor'
        self.PEAK = '_spisslast'
        self.FROM_SOURCE = '_levert_fra_kilde'
        self.DISTRICT_HEATING_PRODUCED = '_fjernvarmeproduksjon'
        self.SOLAR_PANELS_PRODUCED = '_solcelleproduksjon'

#-- functions
def read_arcgis(feature_layer = 'fra_arcgis.xlsx'):
    # read from arcgis -> return df
    df = pd.read_excel(feature_layer)
    return df

#et energiscnarie kan være prosentvis antall av bygg som får ulik energimiks
def create_scenario(df, energy_scenario = """
                    A:F50_V30_S20;
                    B:F20_V30_S20;
                    """):
    constants = Constants()
    # populate df with scenario paramaters
    for supply_technology in [constants.GSHP, constants.SOLAR_PANELS, constants.AIR_SOURCE_HEAT_PUMP, constants.DISTRICT_HEATING]:
        df[supply_technology] = 0
    # based on condition in dataframe
    df[constants.DISTRICT_HEATING] = np.where(df[constants.BUILDING_TYPE] == 'A', 1, 0)
    df[constants.GSHP] = np.where(df[constants.BUILDING_TYPE] == 'B', 1, 0)
    return df

def modify_scenario(df, energy_scenario = "a"):
    #for byggtype A-> fjernvarme 50% varmepumpe30% solceller 20% og oppgradert byggestandard
    constants = Constants()
    # based on condition in dataframe
    df[constants.BUILDING_STANDARD] = np.where(df[constants.BUILDING_STANDARD] == 'X', "Y", df[constants.BUILDING_STANDARD])
    #df[constants.SOLAR_PANELS] = np.where(df[constants.BUILDING_TYPE] == 'A', 1, 0)
    return df

def __get_secret(filename):
    with open(filename) as file:
        secret = file.readline()
    return secret

def __profet_api(building_standard, building_type, area, temperature_array):
    constants = Constants()
    oauth = OAuth2Session(client=BackendApplicationClient(client_id="profet_2023"))
    predict = OAuth2Session(
        token=oauth.fetch_token(
            token_url="https://identity.byggforsk.no/connect/token",
            client_id="profet_2023",
            client_secret=__get_secret("src/secret.txt"),
        )
    )
    selected_standard = constants.BUILDING_STANDARDS[building_standard]
    if selected_standard == "Reg":
        regular_area, efficient_area, veryefficient_area = area, 0, 0
    if selected_standard == "Eff-E":
        regular_area, efficient_area, veryefficient_area = 0, area, 0
    if selected_standard == "Vef":
        regular_area, efficient_area, veryefficient_area = 0, 0, area
    # --
    if temperature_array[0] == 0:
        request_data = {
            "StartDate": "2022-01-01", 
            "Areas": {f"{constants.BUILDING_TYPES[building_type]}": {"Reg": regular_area, "Eff-E": efficient_area, "Eff-N": 0, "Vef": veryefficient_area}},
            "RetInd": False,  # Boolean, if True, individual profiles for each category and efficiency level are returned
            "Country": "Norway"}
    else:
        request_data = {
        "StartDate": "2022-01-01", 
        "Areas": {f"{constants.BUILDING_TYPES[building_type]}": {"Reg": regular_area, "Eff-E": efficient_area, "Eff-N": 0, "Vef": veryefficient_area}},
        "RetInd": False,  # Boolean, if True, individual profiles for each category and efficiency level are returned
        "Country": "Norway",  # Optional, possiblity to get automatic holiday flags from the python holiday library.
        "TimeSeries": {"Tout": temperature_array.tolist()}}
        
    r = predict.post(
        "https://flexibilitysuite.byggforsk.no/api/Profet", json=request_data
    )
    if r.status_code == 200:
        df = pd.DataFrame.from_dict(r.json())
        df.reset_index(drop=True, inplace=True)
        profet_df = df[["Electric", "DHW", "SpaceHeating"]]
        thermal_demand = (profet_df['DHW'] + profet_df['SpaceHeating']).to_numpy()
        electric_demand = profet_df['Electric'].to_numpy()
        return thermal_demand, electric_demand
    else:
        raise TypeError("PROFet virker ikke")

def preprocess_profet_data(df):
    constants = Constants()
    result_data = []
    result_df = pd.DataFrame()
    for building_type in constants.BUILDING_TYPES:
        for building_standard in constants.BUILDING_STANDARDS:
            thermal_demand, electric_demand = __profet_api(building_standard = building_standard, building_type = building_type, area = 1, temperature_array = df[constants.TEMPERATURE_ARRAY][0])
            
            thermal_col_name = f"{building_type}_{building_standard}_THERMAL"
            electric_col_name = f"{building_type}_{building_standard}_ELECTRIC"

            result_df[thermal_col_name] = thermal_demand.flatten()
            result_df[electric_col_name] = electric_demand.flatten()

    result_df.to_csv("src/profet_data.csv", sep = ";")
    return result_df

def profet_calculation_simplified(row):
    constants = Constants()
    thermal_demand_series = constants.PROFET_DATA[f"{row[constants.BUILDING_TYPE]}_{row[constants.BUILDING_STANDARD]}_THERMAL"]
    thermal_demand = row[constants.AREA] * np.array(thermal_demand_series)
    electric_demand_series = constants.PROFET_DATA[f"{row[constants.BUILDING_TYPE]}_{row[constants.BUILDING_STANDARD]}_ELECTRIC"]
    electric_demand = row[constants.AREA] * np.array(electric_demand_series)
    return thermal_demand, electric_demand

def profet_calculation(row):
    constants = Constants()
    # profet API logic
    thermal_demand, electric_demand = __profet_api(building_standard = row[constants.BUILDING_STANDARD], building_type = row[constants.BUILDING_TYPE], area = row[constants.AREA], temperature_array = row[constants.TEMPERATURE_ARRAY])
    return thermal_demand, electric_demand

def __dekningsgrad_calculation(dekningsgrad, timeserie):
        if dekningsgrad == 100:
            return timeserie
        timeserie_sortert = np.sort(timeserie)
        timeserie_sum = np.sum(timeserie)
        timeserie_N = len(timeserie)
        startpunkt = timeserie_N // 2
        i = 0
        avvik = 0.0001
        pm = 2 + avvik
        while abs(pm - 1) > avvik:
            cutoff = timeserie_sortert[startpunkt]
            timeserie_tmp = np.where(timeserie > cutoff, cutoff, timeserie)
            beregnet_dekningsgrad = (np.sum(timeserie_tmp) / timeserie_sum) * 100
            pm = beregnet_dekningsgrad / dekningsgrad
            gammelt_startpunkt = startpunkt
            if pm < 1:
                startpunkt = startpunkt + timeserie_N // 2 ** (i + 2) - 1
            else:
                startpunkt = startpunkt - timeserie_N // 2 ** (i + 2) - 1
            if startpunkt == gammelt_startpunkt:
                break
            i += 1
            if i > 13:
                break
            return timeserie_tmp

def varmepumpe_calculation(row):
    constants = Constants()
    VIRKNINGSGRAD = 1
    kompressor, levert_fra_kilde, spisslast = 0, 0, 0
    #-
    if row[constants.GSHP] == 1 or row[constants.AIR_SOURCE_HEAT_PUMP] == 1:
        if row[constants.GSHP] == 1:
            COEFFICIENT_OF_PERFORMANCES = constants.COEFFICIENT_OF_PERFORMANCES_GSHP
        else:
            COEFFICIENT_OF_PERFORMANCES = constants.COEFFICIENT_OF_PERFORMANCES_ASHP
        varmebehov = row[constants.THERMAL_DEMAND]
        levert_fra_varmepumpe = __dekningsgrad_calculation(constants.DEKNINGSGRADER_GSHP[row[constants.BUILDING_TYPE]], varmebehov * VIRKNINGSGRAD)
        kompressor = levert_fra_varmepumpe / COEFFICIENT_OF_PERFORMANCES[row[constants.BUILDING_TYPE]]
        levert_fra_kilde = levert_fra_varmepumpe - kompressor
        spisslast = varmebehov - levert_fra_varmepumpe
    return kompressor, -levert_fra_kilde, spisslast

def fjernvarme_calculation(row):
    constants = Constants()
    VIRKNINGSGRAD = 1
    DEKNINGSGRAD = 100
    fjernvarme = 0
    if row[constants.DISTRICT_HEATING] == 1:
        fjernvarme = __dekningsgrad_calculation(DEKNINGSGRAD, row[constants.THERMAL_DEMAND] * VIRKNINGSGRAD)
    return -fjernvarme

def solcelle_calculation(row):
    constants = Constants()
    solceller = 0
    if row[constants.SOLAR_PANELS] == 1:
        solceller = row[constants.AREA] * constants.SOLARPANEL_DATA[constants.SOLARPANEL_BUILDINGS[row[constants.BUILDING_TYPE]]].to_numpy()
    return -solceller

def compile_data(row):
    constants = Constants()
    thermal_balance = row[constants.THERMAL_DEMAND] + row[constants.FROM_SOURCE] + row[constants.DISTRICT_HEATING_PRODUCED]
    electric_balance = row[constants.ELECTRIC_DEMAND] + row[constants.COMPRESSOR] + row[constants.PEAK] + row[constants.SOLAR_PANELS_PRODUCED]
    total_balance = thermal_balance + electric_balance
    return total_balance, round(np.sum(total_balance),-2), round(np.max(total_balance),0)

def run_simulation(df, scenario_name, preprocessing = True):
    constants = Constants()
    # profet / demand
    if preprocessing == True:
        df[constants.THERMAL_DEMAND], df[constants.ELECTRIC_DEMAND] = zip(*df.swifter.apply(profet_calculation_simplified, axis=1))
    else:
        df[constants.THERMAL_DEMAND], df[constants.ELECTRIC_DEMAND] = zip(*df.swifter.apply(profet_calculation, axis=1))
    # supply
    df[constants.COMPRESSOR], df[constants.FROM_SOURCE], df[constants.PEAK] = zip(*df.swifter.apply(varmepumpe_calculation, axis=1))
    df[constants.DISTRICT_HEATING_PRODUCED] = df.swifter.apply(fjernvarme_calculation, axis=1)
    df[constants.SOLAR_PANELS_PRODUCED] = df.swifter.apply(solcelle_calculation, axis=1)
    # conclusion
    df[f'_nettutveksling_energi_liste'], df[f'_nettutveksling_energi'], df[f'_nettutveksling_effekt'] = zip(*df.swifter.apply(compile_data, axis=1))
    df.to_csv(f"output/{scenario_name}_unfiltered.csv")
    # cleanup lists
    df.drop([constants.TEMPERATURE_ARRAY, constants.THERMAL_DEMAND, constants.ELECTRIC_DEMAND, constants.COMPRESSOR, constants.FROM_SOURCE, constants.PEAK, constants.DISTRICT_HEATING_PRODUCED, constants.SOLAR_PANELS_PRODUCED, f'_nettutveksling_energi_liste'], axis=1, inplace=True)
    df.to_csv(f"output/{scenario_name}_filtered.csv")
    return df

def to_arcgis(df):
    pass

def add_temperature_series(df, temperature_series = "custom"):
    constants = Constants()
    if temperature_series == "default":
        outdoor_temperature = np.array([0])
    else:
        outdoor_temperature = np.full(24, 6)
    df[constants.TEMPERATURE_ARRAY] = [outdoor_temperature] * df.shape[0]
    return df

def main():
    # -- setup
    original_table = read_arcgis(feature_layer = 'fra_arcgis.xlsx')
    original_table = add_temperature_series(df = original_table, temperature_series = "default")
    table = original_table.copy()
    # -- preprocess profet data
    #profet_data = preprocess_profet_data(df = table)
    # -- simulation 1
    table = create_scenario(df = table)
    table = run_simulation(df = table, scenario_name = "S1")
    # -- simulation 2
    table = modify_scenario(df = table)
    table = run_simulation(df = table, scenario_name = "S2")
    # -- simulation 3
    #to_arcgis(df = table)

# mangler på valg av scenario_konsept
if __name__ == '__main__':
    main()
