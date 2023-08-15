import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient
import swifter
import ast
import random
import os
from arcgis.features import GeoAccessor, GeoSeriesAccessor
import helpscripts.energy_area_ids as energy_area_ids
import pathlib
import arcpy

class EnergyAnalysis:
    def __init__(self):
        self.ENERGY_DICTS = { 
                             "A" : ({
                                 "A" : "S30_O50_F60_G40_V00",
                                 "B" : "F80_G20_V20_S40",
                                 "C" : "F20_V60_S20",
                                 "D" : "F10_V60_S20",
                                 "E" : "F50_V40_S20",
                                 "F" : "F20_V60_S20",
                                 "G" : "F20_V60_S20_O10",
                                 "H" : "F20_V60_S20",
                                 "I" : "F20_V60_S20",
                                 "J" : "F20_V60_S20",
                                 "K" : "F20_V60_S20",
                                 "L" : "F20_V60_S20",
                                 }),
                             "B" : ({
                                 "A" : "S20_O50_F60_G40_V00",
                                 "B" : "F40_G20_V20_S40",
                                 "C" : "F20_V60_S20",
                                 "D" : "F10_V60_S20",
                                 "E" : "F50_V40_S20",
                                 "F" : "F20_V60_S20",
                                 "G" : "F20_V60_S20_O10",
                                 "H" : "F20_V60_S20",
                                 "I" : "F20_V60_S20",
                                 "J" : "F20_V60_S20",
                                 "K" : "F20_V60_S20",
                                 "L" : "F20_V60_S20",
                                 }),
                             "C" : ({
                                 "A" : "S99_O50_F60_G40_V00",
                                 "B" : "F40_G20_V20_S40",
                                 "C" : "F20_V60_S20",
                                 "D" : "F10_V60_S20",
                                 "E" : "F50_V40_S20",
                                 "F" : "F20_V60_S20",
                                 "G" : "F20_V60_S20_O10",
                                 "H" : "F20_V60_S20",
                                 "I" : "F20_V60_S20",
                                 "J" : "F20_V60_S20",
                                 "K" : "F20_V60_S20",
                                 "L" : "F20_V60_S20",
                                 }),
                             "D" : ({
                                 "A" : "S20_O50_F60_G40_V00",
                                 "B" : "F40_G20_V20_S40",
                                 "C" : "F20_V60_S20",
                                 "D" : "F10_V60_S20",
                                 "E" : "F99_V40_S20",
                                 "F" : "F20_V60_S20",
                                 "G" : "F20_V60_S20_O10",
                                 "H" : "F20_V60_S20",
                                 "I" : "F20_V60_S20",
                                 "J" : "F20_V60_S20",
                                 "K" : "F20_V60_S20",
                                 "L" : "F20_V60_S20",
                                 })
                             }
        
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
            'H' : 'Næringsbygg_mindre',
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
        #-- fra matrikkel
        self.OBJECT_ID = 'OBJECTID'
        self.BUILDING_STANDARD = 'Energistandard_profet'
        self.BUILDING_TYPE = 'Byggtype_profet'
        self.BUILDING_AREA = 'BRUKSAREAL_TOTALT'
        self.NUMBER_OF_FLOORS = 'ANTALL_ETASJER'
        self.BUILT_AREA = 'BEBYGD_AREAL'
        self.HAS_WELL = 'Energibronn'
        self.HAS_FJERNVARME = 'Fjernvarme'
        self.ENERGY_AREA_ID = 'Energiomraadeid'
        #-- fra matrikkel
        self.TEMPERATURE_ARRAY = '_utetemperatur'
        self.THERMAL_DEMAND = '_termisk_energibehov'
        self.ELECTRIC_DEMAND = '_elspesifikt_energibehov'
        self.COMPRESSOR = '_kompressor'
        self.PEAK = '_spisslast'
        self.FROM_SOURCE = '_levert_fra_kilde'
        self.DISTRICT_HEATING_PRODUCED = '_fjernvarmeproduksjon'
        self.SOLAR_PANELS_PRODUCED = '_solcelleproduksjon'

    #-- functions
    def read_from_arcgis(self, gdb, rootfolder, feature_class_name):
        #--
        # read from arcgis -> return df
        featureclass_input_path = os.path.join(gdb, feature_class_name)
        energy_area_ids.spatial_join(byggpunkt_fc = featureclass_input_path)
        spatial_df = pd.DataFrame.spatial.from_featureclass(featureclass_input_path)
        return spatial_df
    
    def export_to_arcgis(self, df, gdb, scenario_name = "default"):
        featureclass_output_name = gdb / f"results_{scenario_name}"
        df.spatial.to_featureclass(location=os.path.join(gdb, featureclass_output_name))
    
    def read_excel(self, sheet = 'input/fra_arcgis.xlsx'):
        # read from excel -> return df
        df = pd.read_excel(sheet)
        return df

    def __string_percentage(self, string):
        return int(f"{string[1]}{string[2]}")
    
    def __add_values_randomly(self, df, column_name, percentage, fill_value=True):
        number_of_rows = len(df)
        n_values = int((percentage / 100) * number_of_rows)
        random_indices = random.sample(range(number_of_rows), n_values)
        random_values = [fill_value for _ in range(n_values)]
        df = df.sort_values(self.OBJECT_ID).reset_index(drop=True)
        df.loc[random_indices, column_name] = random_values
        return df
    
    def __add_values_grunnvarme(self, df, fill_value = True):
        df.loc[df[self.HAS_WELL] <= 1, self.GSHP] = fill_value
        return df
    
    def __add_values_fjernvarme(self, df, fill_value = True):
        df.loc[df[self.HAS_FJERNVARME] <= 1, self.DISTRICT_HEATING] = fill_value
        return df
    
    def _add_values_randomly_thermal(self, df, percentages, fill_value=True):
        p1, p2, p3 = (
            percentages[self.GSHP],
            percentages[self.DISTRICT_HEATING],
            percentages[self.AIR_SOURCE_HEAT_PUMP],
        )
        c1, c2, c3 = (
            self.GSHP,
            self.DISTRICT_HEATING,
            self.AIR_SOURCE_HEAT_PUMP,
        )
        # --
        df1 = self.__add_values_randomly(
            df=df, column_name=c1, percentage=p1, fill_value=fill_value
        )

        df2 = df1[df1[c1] != fill_value].reset_index(drop=True)
        df2 = self.__add_values_randomly(
            df=df2, column_name=c2, percentage=p2, fill_value=fill_value
        )

        df3 = df2[df2[c2] != fill_value].reset_index(drop=True)
        df3 = self.__add_values_randomly(
            df=df3, column_name=c3, percentage=p3, fill_value=fill_value
        )

        # slå sammen df1 og df2
        merged_df = pd.concat([df1, df2]).reset_index(drop=True)
        merged_df = merged_df.drop_duplicates(subset=self.OBJECT_ID, keep="last")
        merged_df = merged_df.sort_values(self.OBJECT_ID).reset_index(drop=True)

        # slå sammen (df1 og df2) med df3
        merged_df = pd.concat([merged_df, df3]).reset_index(drop=True)
        merged_df = merged_df.drop_duplicates(subset=self.OBJECT_ID, keep="last")
        merged_df = merged_df.sort_values(self.OBJECT_ID).reset_index(drop=True)
        

        # skriv tilbake df i self
        return merged_df
    
    #et energiscnarie kan være prosentvis antall av bygg som får ulik energimiks
    def __create_scenario(self, df, energy_scenario):
        fill_value = True
        no_fill_value = 0
        # populate df with scenario paramaters
        for supply_technology in [self.GSHP, self.SOLAR_PANELS, self.AIR_SOURCE_HEAT_PUMP, self.DISTRICT_HEATING]:
            df[supply_technology] = 0
        # go through energy_scenario dict
        modified_df_list = []
        for building_type in energy_scenario.keys():
            df_building_type = df.loc[df[self.BUILDING_TYPE] == building_type]
            for supply_type in energy_scenario[building_type].split("_"):
                # case fjernvarme
                if "F" in supply_type:
                    percentage_fjernvarme = self.__string_percentage(string = supply_type)
                # case grunnvarme
                if "G" in supply_type:
                    percentage_grunnvarme = self.__string_percentage(string = supply_type)
                # case varmepumpe
                if "V" in supply_type:
                    percentage_varmepumpe = self.__string_percentage(string = supply_type)
                # case solenergi
                if "S" in supply_type:
                    percentage_solceller = self.__string_percentage(string = supply_type)
                # case oppgradert byggestandard:    
                if "O" in supply_type:
                    percentage_oppgradert = self.__string_percentage(string = supply_type) 
            percentages = {
                self.GSHP : percentage_grunnvarme,
                self.DISTRICT_HEATING : percentage_fjernvarme,
                self.AIR_SOURCE_HEAT_PUMP : percentage_varmepumpe,
            }
            modified_df = self.__add_values_grunnvarme(df = df_building_type, fill_value = fill_value)
            #modified_df = self.__add_values_fjernvarme(df = df_building_type, fill_value = fill_value)
            modified_df = self._add_values_randomly_thermal(df = df_building_type, percentages = percentages)
            modified_df = self.__add_values_randomly(df = modified_df, column_name = self.SOLAR_PANELS, percentage = percentage_solceller, fill_value = fill_value )
            modified_df_list.append(modified_df)
        #-- Merge alle dataframes i områdeliste
        modified_df = pd.concat(modified_df_list).reset_index(drop=True)
        modified_df = modified_df.sort_values(self.OBJECT_ID).reset_index(drop=True)
        return modified_df
    
    def create_scenario(self, df):
        table_splitted_list = []
        for energy_area in df[self.ENERGY_AREA_ID].unique():
            if isinstance(energy_area, str):
                energy_dict = self.ENERGY_DICTS[energy_area]
                table_splitted = df.loc[df[self.ENERGY_AREA_ID] == energy_area]
                table_splitted_list.append(table_splitted)
        df = pd.concat(table_splitted_list).reset_index(drop = True)
        df = df.sort_values(self.OBJECT_ID).reset_index(drop=True)
        #--
        df = self.__create_scenario(df = df, energy_scenario = energy_dict)
        return df

    def modify_scenario(self, df):
        # based on condition in dataframe
        df[self.BUILDING_STANDARD] = np.where(df[self.BUILDING_STANDARD] == 'X', "Y", df[self.BUILDING_STANDARD])
        #df[self.SOLAR_PANELS] = np.where(df[self.BUILDING_TYPE] == 'A', 1, 0)
        return df

    def __get_secret(self, filename):
        with open(filename) as file:
            secret = file.readline()
        return secret

    def __profet_api(self, building_standard, building_type, area, temperature_array):
        oauth = OAuth2Session(client=BackendApplicationClient(client_id="profet_2023"))
        predict = OAuth2Session(
            token=oauth.fetch_token(
                token_url="https://identity.byggforsk.no/connect/token",
                client_id="profet_2023",
                client_secret=self.__get_secret("src/secret.txt"),
            )
        )
        selected_standard = self.BUILDING_STANDARDS[building_standard]
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
                "Areas": {f"{self.BUILDING_TYPES[building_type]}": {"Reg": regular_area, "Eff-E": efficient_area, "Eff-N": 0, "Vef": veryefficient_area}},
                "RetInd": False,  # Boolean, if True, individual profiles for each category and efficiency level are returned
                "Country": "Norway"}
        else:
            request_data = {
            "StartDate": "2022-01-01", 
            "Areas": {f"{self.BUILDING_TYPES[building_type]}": {"Reg": regular_area, "Eff-E": efficient_area, "Eff-N": 0, "Vef": veryefficient_area}},
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

    def preprocess_profet_data(self, df):
        result_data = []
        result_df = pd.DataFrame()
        for building_type in self.BUILDING_TYPES:
            for building_standard in self.BUILDING_STANDARDS:
                thermal_demand, electric_demand = self.__profet_api(building_standard = building_standard, building_type = building_type, area = 1, temperature_array = df[self.TEMPERATURE_ARRAY][0])
                
                thermal_col_name = f"{building_type}_{building_standard}_THERMAL"
                electric_col_name = f"{building_type}_{building_standard}_ELECTRIC"

                result_df[thermal_col_name] = thermal_demand.flatten()
                result_df[electric_col_name] = electric_demand.flatten()

        result_df.to_csv("src/profet_data.csv", sep = ";")
        return result_df

    def profet_calculation_simplified(self, row):
        thermal_demand_series = self.PROFET_DATA[f"{row[self.BUILDING_TYPE]}_{row[self.BUILDING_STANDARD]}_THERMAL"]
        thermal_demand = row[self.BUILDING_AREA] * np.array(thermal_demand_series)
        electric_demand_series = self.PROFET_DATA[f"{row[self.BUILDING_TYPE]}_{row[self.BUILDING_STANDARD]}_ELECTRIC"]
        electric_demand = row[self.BUILDING_AREA] * np.array(electric_demand_series)
        return thermal_demand, electric_demand

    def profet_calculation(self, row):
        # profet API logic
        thermal_demand, electric_demand = self.__profet_api(building_standard = row[self.BUILDING_STANDARD], building_type = row[self.BUILDING_TYPE], area = row[self.BUILDING_AREA], temperature_array = row[self.TEMPERATURE_ARRAY])
        return thermal_demand, electric_demand

    def __dekningsgrad_calculation(self, dekningsgrad, timeserie):
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

    def varmepumpe_calculation(self, row):
        VIRKNINGSGRAD = 1
        kompressor, levert_fra_kilde, spisslast = 0, 0, 0
        #-
        if row[self.GSHP] == 1 or row[self.AIR_SOURCE_HEAT_PUMP] == 1:
            if row[self.GSHP] == 1:
                COEFFICIENT_OF_PERFORMANCES = self.COEFFICIENT_OF_PERFORMANCES_GSHP
            else:
                COEFFICIENT_OF_PERFORMANCES = self.COEFFICIENT_OF_PERFORMANCES_ASHP
            varmebehov = row[self.THERMAL_DEMAND]
            levert_fra_varmepumpe = self.__dekningsgrad_calculation(self.DEKNINGSGRADER_GSHP[row[self.BUILDING_TYPE]], varmebehov * VIRKNINGSGRAD)
            kompressor = levert_fra_varmepumpe / COEFFICIENT_OF_PERFORMANCES[row[self.BUILDING_TYPE]]
            levert_fra_kilde = levert_fra_varmepumpe - kompressor
            spisslast = varmebehov - levert_fra_varmepumpe
        return kompressor, -levert_fra_kilde, spisslast

    def fjernvarme_calculation(self, row):
        VIRKNINGSGRAD = 1
        DEKNINGSGRAD = 100
        fjernvarme = 0
        if row[self.DISTRICT_HEATING] == 1:
            fjernvarme = self.__dekningsgrad_calculation(DEKNINGSGRAD, row[self.THERMAL_DEMAND] * VIRKNINGSGRAD)
        return -fjernvarme

    def solcelle_calculation(self, row):
        solceller = 0
        if row[self.BUILDING_AREA] != 0:
            area = row[self.BUILT_AREA]
            if area == 0:    
                area = row[self.BUILDING_AREA] / row[self.NUMBER_OF_FLOORS]
            if row[self.SOLAR_PANELS] == 1:
                solceller = area * self.SOLARPANEL_DATA[self.SOLARPANEL_BUILDINGS[row[self.BUILDING_TYPE]]].to_numpy()
        return -solceller

    def compile_data(self, row):
        thermal_balance = row[self.THERMAL_DEMAND] + row[self.FROM_SOURCE] + row[self.DISTRICT_HEATING_PRODUCED]
        electric_balance = row[self.ELECTRIC_DEMAND] + row[self.COMPRESSOR] + row[self.PEAK] + row[self.SOLAR_PANELS_PRODUCED]
        total_balance = thermal_balance + electric_balance
        return total_balance, round(np.sum(total_balance),-2), round(np.max(total_balance),0)
    
    def chunkify(self, df, chunk_size):
        list_df = [df[i:i+chunk_size] for i in range(0,df.shape[0],chunk_size)]
        return list_df

    def run_simulation(self, df, scenario_name, preprocessing = True, chunk_size = 100, test = True):
        def __merge_dataframe_list(df_chunked_list):
            df_results = pd.concat(df_chunked_list).reset_index(drop=True)
            df_results = df_results.sort_values(self.OBJECT_ID).reset_index(drop=True)
            df_results.to_csv(f"output/{scenario_name}_filtered.csv")
            return df_results
            
        df_chunked_list = []
        chunked = self.chunkify(df = df, chunk_size = chunk_size)
        for index, df_chunked in enumerate(chunked):
            # profet / demand
            if preprocessing == True:
                df_chunked[self.THERMAL_DEMAND], df_chunked[self.ELECTRIC_DEMAND] = zip(*df_chunked.swifter.apply(self.profet_calculation_simplified, axis=1))
            else:
                df_chunked[self.THERMAL_DEMAND], df_chunked[self.ELECTRIC_DEMAND] = zip(*df_chunked.swifter.apply(self.profet_calculation, axis=1))
            # supply
            df_chunked[self.COMPRESSOR], df_chunked[self.FROM_SOURCE], df_chunked[self.PEAK] = zip(*df_chunked.swifter.apply(self.varmepumpe_calculation, axis=1))
            df_chunked[self.DISTRICT_HEATING_PRODUCED] = df_chunked.swifter.apply(self.fjernvarme_calculation, axis=1)
            df_chunked[self.SOLAR_PANELS_PRODUCED] = df_chunked.swifter.apply(self.solcelle_calculation, axis=1)
            # conclusion
            df_chunked[f'_nettutveksling_energi_liste'], df_chunked[f'_nettutveksling_energi'], df_chunked[f'_nettutveksling_effekt'] = zip(*df_chunked.swifter.apply(self.compile_data, axis=1))
            df_chunked.to_csv(f"output/{scenario_name}_unfiltered.csv")
            # cleanup lists
            df_chunked.drop([self.TEMPERATURE_ARRAY, self.THERMAL_DEMAND, self.ELECTRIC_DEMAND, self.COMPRESSOR, self.FROM_SOURCE, self.PEAK, self.DISTRICT_HEATING_PRODUCED, self.SOLAR_PANELS_PRODUCED, f'_nettutveksling_energi_liste'], axis=1, inplace=True)
            # append to
            df_chunked_list.append(df_chunked)
            # write more often, for every 5 chunk
            if (index % 5) == 0:
                __merge_dataframe_list(df_chunked_list)
            if index == 10 and test == True:
                break 
        df_results = __merge_dataframe_list(df_chunked_list)
        return df_results
    
    def add_temperature_series(self, df, temperature_series = "custom"):
        if temperature_series == "default":
            outdoor_temperature = np.array([0])
        else:
            outdoor_temperature = np.full(24, 6)
        df[self.TEMPERATURE_ARRAY] = [outdoor_temperature] * df.shape[0]
        return df

    def main(self):
        runner = "magne.syljuasen"
        rootfolder = pathlib.Path(r'C:\Users\magne.syljuasen\Downloads\GIS\GIS'.format(runner))
        gdb = rootfolder / 'Datagrunnlag.gdb'
        featureclass_input_name = gdb / "Byggpunkt_040623_vasket"
        # -- setup
        #table = self.read_excel(feature_layer = 'input/fra_arcgis.xlsx')
        table = self.read_from_arcgis(gdb = gdb, rootfolder = rootfolder, feature_class_name = featureclass_input_name)
        # -- preprocess profet data
        #profet_data = preprocess_profet_data(df = table)
        # -- simulation 1
        table = self.create_scenario(df = table)
        table = self.add_temperature_series(df = table, temperature_series = "default")
        table = self.run_simulation(df = table, scenario_name = "S1", test = True)
        self.export_to_arcgis(df = table, gdb = gdb, scenario_name = "default")   
        # -- simulation 2
        table = self.modify_scenario(df = table)
        table = self.add_temperature_series(df = table, temperature_series = "default")
        table = self.run_simulation(df = table, scenario_name = "S2", test = True)
        # -- simulation 3
        self.export_to_arcgis(df = table, gdb = gdb, scenario_name = "s1")   
        
        
if __name__ == '__main__':
    runner= 'torbjorn.boe'
    rootfolder = pathlib.Path(r'C:\Users\magne.syljuasen\Downloads\GIS\GIS'.format(runner))
    gdb = rootfolder / 'Datagrunnlag.gdb'
    output_fc_name = "Test_Energianlyse_TEB"
    byggpunkt_fc_name = gdb / "Byggpunkt_040623_vasket"
    omraadeidfelt='Omraadeid'
    # Logg settings
    #logfile = rootfolder / 'Energianalyselog_Zero.log'
    #Loggit.OpprettLogg(filename=logfile, folder=rootfolder)
    #logger = logging.getLogger('Energianalyselog')
    #logger.info(f'Parametrer: rotmappe {rootfolder}, out fc {output_fc_name}, gdb {gdb.name}, byggpunkt {byggpunkt_fc_name.name}')
    #--
    #scenario='dagens_situasjon'
    #test= True
    
    EnergyAnalysis().main()
