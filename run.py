import requests
import geopandas as gpd
import yaml
import os
import paramiko


def ScenceFinder(shape_path, satellite, processing_level, product_type, start_date, end_date, cloud_cover, output_format='json', maxRecords = 10):

    start_date = start_date.replace(":", "%3A") 
    end_date = end_date.replace(":", "%3A") 

    shape = gpd.read_file(shape_path)
    shape_4326 = shape.to_crs(epsg=4326)
    
    wkt_list = list(shape_4326['geometry'][0].exterior.coords)

    s = ""

    for idx, point in enumerate(wkt_list):
        s = s + str(point[0]) + "+" + str(point[1])
        if idx != len(wkt_list)-1:
            s = s + "%2C"
        else:
            pass

    geometry = f"POLYGON(({s}))"        

    base_url = "http://finder.code-de.org/resto/api/collections/"

    if satellite == "Sentinel1":
        modified_url = f"{satellite}/search.{output_format}?{maxRecords}&startDate={start_date}&completionDate={end_date}&location=all&processingLevel={processing_level}&productType={product_type}&sortParam=startDate&sortOrder=descending&status=all&geometry={geometry}&dataset=ESA-DATASET"
    else:
        modified_url = f"{satellite}/search.{output_format}?{maxRecords}&startDate={start_date}&completionDate={end_date}&cloudCover=[0%2C{cloud_cover}]&location=all&processingLevel={processing_level}&productType={product_type}&sortParam=startDate&sortOrder=descending&status=all&geometry={geometry}&dataset=ESA-DATASET"
    
    url = base_url + modified_url

    print(url)

    resp = requests.get(url).json()

    return resp


if os.path.exists("config.yaml"):
    with open('config.yaml') as f:
        
        data = yaml.load(f, Loader=yaml.FullLoader)

        shape_path = data['classification']['AOI']['shapefile']['path']
        satellite = data['classification']['data_source']['satellite']['name']
        processing_level = data['classification']['data_source']['satellite']['processing_level']
        product_type = data['classification']['data_source']['satellite']['product_type']
        start_date = data['classification']['data_source']['satellite']['start_date']
        end_date = data['classification']['data_source']['satellite']['end_date']
        cloud_cover = data['classification']['data_source']['satellite']['cloud_cover']


    
scenes = ScenceFinder(shape_path, satellite, processing_level, product_type, start_date, end_date, cloud_cover)

list_path = []

for feature in scenes['features']:
    list_path.append(feature['properties']['productIdentifier'])

for i in list_path:
    print(i)

"""
import os
import paramiko
paramiko.util.log_to_file('/tmp/paramiko.log')
paramiko.util.load_host_keys(os.path.expanduser('~/.ssh/known_hosts'))

host = 'local'
port = 22
username = 'user'

files = ['file1', 'file2', 'file3', 'file4']
remote_images_path = '/remote_path/images/'
local_path = '/tmp/'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(
            paramiko.AutoAddPolicy())
ssh.connect(hostname=host, port=port, username=username)
sftp = ssh.open_sftp()

for file in files:
    file_remote = remote_images_path + file
    file_local = local_path + file

    print file_remote + '>>>' + file_local

    sftp.get(file_remote, file_local)

sftp.close()
ssh.close()



"""







