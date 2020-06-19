
def download_file(url, local_filename):
    """From https://stackoverflow.com/questions/16694907/"""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
    return local_filename

def get_pt_data():
    URL = "https://raw.githubusercontent.com/beoutbreakprepared/nCoV2019/master/latest_data/latestdata.tar.gz"
    LINELIST_PATH = 'data/linelist.tar.gz'

    if not os.path.exists(LINELIST_PATH):
        print('Downloading file, this will take a while ~100mb')
        try:
            download_file(URL, LINELIST_PATH)
            clear_output(wait=True)
            print('Done downloading.')
        except:
            print('Something went wrong. Try again.')
    else:
        print('Already downloaded CSV')
