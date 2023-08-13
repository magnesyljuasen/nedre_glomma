
        
    # renaming
    df.columns = [col + f'_{scenario_name}' if col.startswith('_') and '__' not in col else col for col in df.columns]
    df = df.assign(
        **{f"{constants.GSHP}": df[constants.GSHP]}, 
        **{f"{constants.SOLAR_PANELS}": df[constants.SOLAR_PANELS]}, 
        **{f"{constants.AIR_SOURCE_HEAT_PUMP}": df[constants.AIR_SOURCE_HEAT_PUMP]}, 
        **{f"{constants.DISTRICT_HEATING}": df[constants.DISTRICT_HEATING]})