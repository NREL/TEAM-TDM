import requests
import zipfile
import sys
import numpy as np
from collections.abc import Iterable

def duplicate_columns(frame):
    ''' This function returns which columns in a pandas 
        dataframe contain identical data to some other column '''
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break

    return dups

def download_a_thing(url, local_filename):
    ''' This function downloads a thing from url to a local_filename '''
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                #f.flush() commented by recommendation from J.F.Sebastian
    return local_filename

def unzip_a_thing(path_to_zip_file, directory_to_extract_to):
    ''' This function unzips a zip file.  Useful for auto-downloading/loading datasets '''
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()
    return directory_to_extract_to  
        
def flatten(items):
    """Yield items from any nested iterable;
       list(flatten([[1,2],[[[3]]]])) == [1,2,3] """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x
            
def integerify(x):
    ''' make a float an integer '''
    return np.round(x).astype(np.int64)
    
def cmap_one(i):
    return "#"+hex(((int(i)+1)*2396745)%(256**3))[2:].rjust(6,"0")
def cmap(colors):
    ''' This function produces sequential html colors that are "far" from one another.
        Handy for getting "very different" colors for classes in a graph '''
    return list(map(cmap_one, colors))