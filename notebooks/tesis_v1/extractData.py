#%%
import pandas as pd
import os
from sodapy import Socrata

def get_path(folder, file_path):
    current_directory = os.getcwd()
    return os.path.join(current_directory, folder, file_path)

def get_query(folder, file_path):
    path = get_path(folder, file_path)
    # try to get the query
    with open(path, "r", encoding="utf8") as query_file:
        query = query_file.read()
    return query

def parse_to_list(ls):
    ids = [ str(i) for i in ls]
    ids = "'" + "','".join(ids) + "'"
    return ids

def querySecop(query, url='www.datos.gov.co', id_data='f789-7hwg', api_key='SOCRATA_APP_TOKEN', timeout=1000):
    """
    Extract data from the Socrata API
    """

    # Create client
    socrata_token = os.environ.get(api_key)

    client = Socrata(url, 
                     socrata_token, 
                     username=os.environ.get("SOCRATA_USERNAME"),
                     password=os.environ.get("SOCRATA_PASSWORD"),
                     timeout=timeout)


    # Query data
    query_results = client.get(id_data, query=query)
    query_results = pd.DataFrame.from_dict(query_results)
    print("El numero de contratos extraidos: {}".format(query_results.shape[0]))
    return query_results



#%% Link to open contracting data
SECOPI_PROCESS_API = 'f789-7hwg'
SECOPII_CONTRATOS_API = 'jbjy-vk9h'
SECOPI_ADDITIONS_API = '7fix-nd37'
SECOPI_PUNISHMENT_API = '4n4q-k399'


#%% Request SECOP I Procesos
queryPath = get_path('', 'requestSECOPIProcesos.sql')
querySECOPI = get_query('', queryPath)
#print(querySECOPI)
procesosSECOPI = querySecop(querySECOPI, url='www.datos.gov.co', id_data=SECOPI_PROCESS_API, api_key='SOCRATA_APP_TOKEN', timeout=1000)

#%%
procesosSECOPI.to_csv('procesosSECOPI.csv', index=False)
procesosSECOPI.to_excel('procesosSECOPI.xlsx', index=False)

#%% Request SECOP II Contratos Electrónicos
queryPath = get_path('', 'requestSECOPIIContratos.sql')
querySECOPIIContratos = get_query('', queryPath)
contratosSECOPII = querySecop(querySECOPIIContratos, url='www.datos.gov.co', id_data=SECOPII_CONTRATOS_API, api_key='SOCRATA_APP_TOKEN', timeout=1000)

# %%
contratosSECOPII.to_csv('contratosSECOPII.csv', index=False)
contratosSECOPII.to_excel('contratosSECOPII.xlsx', index=False)
# %%
