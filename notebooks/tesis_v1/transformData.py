import pandas as pd
#import ast
import numpy as np
import unidecode



def dataCleaning(df):
    """
    Process and clean the dataset for rural road projects in Colombia.
    """

    # Mapping of Spanish column names to English
    column_translation = {
        'uid': 'uid',
        'estado_del_proceso':'tenderStatus',
        'nombre_de_la_entidad': 'buyerName',
        'nit_de_la_entidad': 'buyerId',
        'departamento_entidad': 'buyerDepartment',
        'municipio_entidad': 'buyerLocation',
        'orden_entidad': 'buyerLevel',
        'modalidad': 'contractMethodCountryLaw',
        'causal_contratacion_directa': 'direct_contracting_cause',
        'objeto_a_contratar': 'contractCategory',
        'tipo_contrato':'contractType',
        'detalle_objeto': 'contractDescription',
        'cuantia_proceso': 'tenderValue',
        'cuantia_contrato': 'contractValue',
        'valor_total_de_adiciones': 'contractAditionalValue',
        'valor_contrato_con_adiciones': 'contractTotalValue',
        'anno_firma': 'contractYearSigned',
        'fecha_firma': 'contractDateSigned',
        'fecha_ini_ejec_contrato': 'contractStartDate',
        'plazo_de_ejec_del_contrato': 'contractDuration',
        'rango_de_ejec_del_contrato': 'contractDurationRange',
        'tiempo_adiciones_en_dias': 'additionalTimeDays',
        'tiempo_adiciones_en_meses': 'additionalTimeMonths',
        'fecha_fin_ejec_contrato': 'contractEndDate',
        'id_adjudicacion': 'awardId',
        'urlproceso': 'urlTender',
        'identificacion_del_contratista': 'supplierId',
        'nom_razon_social_contratista': 'supplierName',
        'dpto_y_muni_contratista': 'supplierLocation',
        'departamento_proveedor': 'supplierDepartment',
        'base_de_datos': 'databaseSource'
    }

    # Apply the translation to the DataFrame
    df.rename(columns=column_translation, inplace=True)

    # Set column data types
    df = df.astype({
        'tenderValue': 'float',
        'contractValue': 'float',
        'contractAditionalValue': 'float',
        'contractTotalValue': 'float',
        'contractYearSigned': 'int32',
        'contractDuration': 'int32',
        'additionalTimeDays': 'int32',
        'additionalTimeMonths': 'int32',
        'contractDateSigned': 'datetime64[ns]',
        'contractStartDate': 'datetime64[ns]',
        'contractEndDate': 'datetime64[ns]',
        'awardId': 'object',
        'buyerId': 'object',
        'supplierId': 'object',
        'supplierLocation': 'object'
    })

    # Correct errors in the contracts values
    corrections = pd.read_excel("dataCorrections.xlsx", sheet_name='contractValues')
    for _, row in corrections.iterrows():
        contract_id = row['uid']
        df.loc[df['uid'] == contract_id, row.index[1:]] = row.values[1:]

    # Remove specific contracts manually
    uid_to_remove = [
        '17-12-7312505-6645201',  # Agreement
        '17-12-7222328-6566950',  # Agreement
        '15-12-4210058-3886615',  # Agreement
        '18-4-8537252-7766100',   # Agreement
        '15-12-3977166-3689414',  # Agreement
        '15-12-3977324-3689541',
        '17-12-7300265-6634055',
        '19-12-9053791-8235792',
        '18-12-8560016-7790587',
        '18-12-8778695-7979822',
        '19-4-10209011-9336255'  # Co-financing
    ]
    df = df.loc[~df['uid'].isin(uid_to_remove)]

    # Scale columns associated with values in minimum wages
    valueColumns = ['tenderValue', 'contractValue', 'contractAditionalValue', 'contractTotalValue']

    minimum_wage = {
        2014: 616000,
        2015: 644350,
        2016: 689455,
        2017: 737717,
        2018: 781242,
        2019: 828116,
        2020: 877803,
        2021: 908526,
        2022: 1000000,
        2023: 1160000,
        2024: 1300000
    }
    # Filter data according contractYearSigned
    df = df.loc[df['contractYearSigned'].isin(list(minimum_wage.keys()))]
    
    # Create a column with the month of each date of the contract
    for col in ['contractDateSigned', 'contractStartDate', 'contractEndDate']:
        colMonth = col.replace('Date','Month')
        df[colMonth] = df[col].dt.month

    for col in valueColumns:
        df[f'{col}Mw'] = df.apply(lambda x: x[col] / minimum_wage[x['contractYearSigned']], axis=1)

    # Convert contract duration to days
    df.loc[df['contractDurationRange'] == 'M', 'contractDuration'] *= 30
    df.drop(columns=['contractDurationRange'], inplace=True)

    # Unify additional contract duration
    df['contractAditionalDuration'] = df['additionalTimeDays'] + df['additionalTimeMonths'] * 30
    df.drop(columns=['additionalTimeMonths','additionalTimeDays'], inplace=True)

    # Calculate final project duration
    df['contractTotalDuration'] = df['contractDuration'] + df['contractAditionalDuration']

    # Correct errors in duration
    corrections = pd.read_excel("dataCorrections.xlsx", sheet_name='contractDurations')
    for _, row in corrections.iterrows():
        contract_id = row['uid']
        df.loc[df['uid'] == contract_id, row.index[1:]] = row.values[1:]

    # Add deviation indicators
    df['haveCostDeviation'] = (df['contractAditionalValue'] > 0).astype(int)
    df['haveTimeDeviation'] = (df['contractAditionalDuration'] > 0).astype(int)
    df['haveTimeAndCostDeviation'] = ((df['contractAditionalValue'] > 0) & (df['contractAditionalDuration'] > 0)).astype(int)

    # Calculate project intensity
    df['projectIntensity'] = df['contractValue'] / df['contractDuration']
    df['projectIntensityMw'] = df['contractValueMw'] / df['contractDuration']

    # Calculate award growth
    df['awardGrowth'] = (df['contractValue'] - df['tenderValue']) / df['tenderValue']

    # Calculate cost deviation
    df['costDeviationPerc'] = (df['contractTotalValue'] - df['contractValue']) / df['contractValue']

    # Calculate time deviation
    df['timeDeviationPerc'] = (df['contractTotalDuration'] - df['contractDuration']) / df['contractDuration']

    # Define territorial entity and responsible party
    #territorial_entity_mapping = {
    #    'TERRITORIAL DISTRITAL MUNICIPAL NIVEL 1': 'Type 1',
    #    'TERRITORIAL DISTRITAL MUNICIPAL NIVEL 2': 'Type 2',
    #    'TERRITORIAL DISTRITAL MUNICIPAL NIVEL 3': 'Type 3',
    #    'TERRITORIAL DISTRITAL MUNICIPAL NIVEL 4': 'Type 4',
    #    'TERRITORIAL DISTRITAL MUNICIPAL NIVEL 5': 'Type 5',
    #    'TERRITORIAL DISTRITAL MUNICIPAL NIVEL 6': 'Type 6',
    #    'TERRITORIAL DEPARTAMENTAL CENTRALIZADO': 'Deparment centralized',
    #    'TERRITORIAL DEPARTAMENTAL DESCENTRALIZADO': 'Deparment decentralized',
    #    'DISTRITO CAPITAL': 'Capital Distric',
    #    'NACIONAL CENTRALIZADO': 'National Centralized',
    #    'NACIONAL DESCENTRALIZADO': 'National Decentralized',
    #    'No Definido':'Not Defined'
    #}

    #df['buyerLevel'] = df['buyerLevel'].replace(territorial_entity_mapping)

    #responsible_party_mapping = {
    #    'Type 1': 'Local Government',
    #    'Type 2': 'Local Government',
    #    'Type 3': 'Local Government',
    #    'Type 4': 'Local Government',
    #    'Type 5': 'Local Government',
    #    'Type 6': 'Local Government',
    #    'Deparment centralized': 'Department',
    #    'Deparment decentralized': 'Department',
    #    'Capital Distric': 'Capital District',
    #    'National Centralized': 'National Entity'
    #}

    #df['buyerGovernmentLevel'] = df['buyerLevel'].replace(responsible_party_mapping)
    #df.drop(columns=['buyerLevel'], inplace=True)

    # Assign regions
    AMAZONIA = ['AMAZONAS', 'CAQUETA', 'PUTUMAYO', 'GUAINIA', 'GUAVIARE', 'VAUPES']
    ORINOQUIA = ['META', 'ARAUCA', 'CASANARE', 'VICHADA']
    ANDINA = ['ANTIOQUIA','BOYACA', 'CALDAS', 'CUNDINAMARCA', 'HUILA', 'NORTE DE SANTANDER',
              'QUINDIO','RISARALDA','SANTANDER','TOLIMA','BOGOTA D.C.']
    CARIBE = ['ATLANTICO','BOLIVAR','CESAR','CORDOBA','LA GUAJIRA','MAGDALENA','SUCRE','SAN ANDRES, PROVIDENCIA Y SANTA CATALINA']
    PACIFICA = ['CAUCA', 'VALLE DEL CAUCA', 'CHOCO', 'NARINO']

    df['buyerRegion'] = df['buyerDepartment'].apply(lambda x: 'AMAZONIA' if x in AMAZONIA else
                                            'ORINOQUIA' if x in ORINOQUIA else
                                            'ANDINA' if x in ANDINA else
                                            'CARIBE' if x in CARIBE else
                                            'PACIFICA' if x in PACIFICA else
                                            'OTRA')

    # Categorize contract method
    df['contractMethodCountryLaw'] = df['contractMethodCountryLaw'].apply(lambda x: unidecode.unidecode(x).upper())
    contractMethod = {
        'CONTRATACION DIRECTA (LEY 1150 DE 2007)': 'Closed',
        'REGIMEN ESPECIAL': 'Special Regime',
        'SELECCION ABREVIADA DE MENOR CUANTIA (LEY 1150 DE 2007)': 'Simplified',
        'LICITACION PUBLICA': 'Open',
        'CONTRATACION MINIMA CUANTIA': 'Limited',
        'LICITACION OBRA PUBLICA': 'Open',
        'CONCURSO DE MERITOS ABIERTO': 'Other',
        'SELECCION ABREVIADA DEL LITERAL H DEL NUMERAL 2 DEL ARTICULO 2 DE LA LEY 1150 DE 2007': 'Simplified',
        'CONCURSO DE DISEÑO ARQUITECTONICO': 'Other',
        'CONTRATOS Y CONVENIOS CON MAS DE DOS PARTES': 'Other'
    }

    df['contractMethod'] = df['contractMethodCountryLaw'].replace(contractMethod)

    df = df.loc[~df['contractMethod'].isin(['Other'])].copy()
    contractMethodOrder = ['Open', 'Simplified', 'Limited', 'Closed', 'Special Regime']
    df['contractMethod'] = pd.Categorical(df['contractMethod'], categories=contractMethodOrder, ordered=True)


    #df['urlTender'] = df['urlTender'].apply(lambda x: ast.literal_eval(x)['url'])

    return df


def workCategory(df):
    # Asignar tipo de obra
    df['workCategoryConstruction'] = df['contractDescription'].str.contains('CONSTRUCCION|PAVIMENTACION|COSNTRUCCION|COSNTRUCCION|CONTRUCCION|CONSTRUIR|ADECUACUACION')
    df['workCategoryMaintenance'] = df['contractDescription'].str.contains('MANTENIMIENTO|MANTEMIENTO|MANTENER|MANTEMIENTO|MATENIMIENTO|MANTANIMIENTO|MANTENIIENTO|MATENIMIENTO|MANTANIMIENTO|MANTANIMIENTO|MANTENIIENTO')
    df['workCategoryRehabilitation'] = df['contractDescription'].str.contains('REHABILITACION|REHABILITAR|REHABLITACION|RECUPERACION|REHABLITACION')
    df['workCategoryImprovement'] = df['contractDescription'].str.contains('MEJORAMIENTO|MEJORAR|MEJORAMENTO|MEJORAMAIENTO|MEKJORAMIENTO|MEJORAMAIENTO|MEJORAMENTO|MEKJORAMIENTO|MEJORAMENTO')
    df['totalWorkCategory'] = np.int32(df['workCategoryConstruction'] + df['workCategoryMaintenance'] + df['workCategoryRehabilitation'] + df['workCategoryImprovement'])
    df['workCategoryOthers'] = df['totalWorkCategory'] == 0

    # Lista de columnas de tipos de obra
    workCategory = ['workCategoryConstruction',
                    'workCategoryMaintenance',
                    'workCategoryRehabilitation',
                    'workCategoryImprovement',
                    'workCategoryOthers']

    # Crear columna WORK_TYPE
    df['workCategory'] = ''

    # Asignar los valores correspondientes a WORK_TYPE
    for col in workCategory:
        df.loc[df[col] == True, 'workCategory'] = col.replace('workCategory', '')

    df.drop(columns=workCategory + ['totalWorkCategory'], inplace=True)

    df = df.loc[~df['workCategory'].isin(['Others'])].copy()

    workCategoryOrder = ['Construction', 'Improvement', 'Rehabilitation', 'Maintenance']
    df['workCategory'] = pd.Categorical(df['workCategory'], categories=workCategoryOrder, ordered=True)

    return df
