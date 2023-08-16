# henter energiomraadenavn til byggpunkt.
import arcpy
#import Agol_data

def spatial_join(byggpunkt_fc: str):
    """
    Kalkulerer energiområdetilhørighet for byggpunkt, basert på områder som ligger i Agol.
    :param byggpunkt_fc: string, byggpunkt som featureclass
    :return: str, featureclass
    """
    #credentials = credentials_agol.credetials_json()
    #gis= Agol_data.Agol(username=credentials['username'], password=credentials['password'])
    #energiomraader = gis.get_hostefeaturelayerdata(energiomraader_itemid)
    #energiomraader_featureset= energiomraader.query()
    energiomraader= 'https://services.arcgis.com/whQdER0woF1J7Iqk/arcgis/rest/services/Energiområder/FeatureServer/3'
    fl= arcpy.MakeFeatureLayer_management(in_features=energiomraader, out_layer='energiomr_lyr')
    byggpunkt_energiomraadekeyfield= 'Energiomraadeid'
    arcpy.AddField_management(in_table= byggpunkt_fc, field_name= byggpunkt_energiomraadekeyfield, field_type='TEXT')
    energiomraader_shapes= {row[0]: row[1] for row in arcpy.da.SearchCursor(fl, ['Områdenavn', 'Shape@'])}
    for omraddenavn, omraadeshape in energiomraader_shapes.items():
        print(f'Henter område {omraddenavn} og beregner tilhørighet')
        with arcpy.da.UpdateCursor(byggpunkt_fc, [byggpunkt_energiomraadekeyfield, 'Shape@']) as cur:
            for row in cur:
                if omraadeshape.contains(row[1]):
                    row[0]= omraddenavn
                cur.updateRow(row)

if __name__ == '__main__':
    spatial_join(byggpunkt_fc=r'C:\Users\torbjorn.boe\Asplan Viak\640572-01 Nedre Glomma - potensial reduksjon effekt og energi - Dokumenter\General\GIS\Datagrunnlag.gdb\Byggpunkt_040623_vasket')
