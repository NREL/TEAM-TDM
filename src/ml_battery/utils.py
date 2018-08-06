import requests
import zipfile
import sys
import numpy as np
from collections.abc import Iterable

def duplicate_columns(frame):
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
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                #f.flush() commented by recommendation from J.F.Sebastian
    return local_filename

def unzip_a_thing(path_to_zip_file, directory_to_extract_to):
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()
    return directory_to_extract_to
    
class LogStdout(object):
    def __init__(self, f):
        import sys
        self.f = f
    def __enter__(self):
        self.old_stdout = sys.stdout
        sys.stdout = self.f
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.old_stdout   
        
def flatten(items):
    """Yield items from any nested iterable; see REF."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x
            
def integerify(x):
    return np.round(x).astype(np.int64)
    
def cmap_one(i):
    return "#"+hex(((int(i)+1)*2396745)%(256**3))[2:].rjust(6,"0")
def cmap(colors):
    return list(map(cmap_one, colors))