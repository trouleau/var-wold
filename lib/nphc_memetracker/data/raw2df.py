from multiprocessing import Pool
import pandas as pd
import gzip
import glob
from urllib.parse import urlparse


####################
# Useful functions #
####################

def apply_inplace(df, field, fun):
    return pd.concat([df.drop(field, axis=1), df[field].apply(fun)], axis=1)


def parse_url(url):
    try:
        o = urlparse(url)
        return o.scheme + "://" + o.netloc
    except ValueError:
        #print("pb with url : ",url)
        return url


###################
# Function to map #
###################

def worker(filename):
    """
    Convert raw file into DataFrame
    """
    with gzip.open(filename, 'rb') as f:
        content = [x.decode().strip('\n') for x in f.readlines()]
        df_rows = []
        num_post = -1
        for line in content:
            x = line.rstrip('\n').split('\t')
            if x[0] == 'P':
                num_post += 1
                post_url = parse_url(x[1])
                post_url_full = x[1]
                links = []
            elif x[0] == 'T':
                date = x[1]
            elif x[0] == 'L':
                link = parse_url(x[1])
                link_full = x[1]
                if link not in links:
                    links.append(link)
            elif x[0] == '':
                if len(links) == 0:
                    row = [date, '', post_url, num_post, 1, post_url_full, '']
                    df_rows.append(row)
                else:
                    for link in links:
                        row = [date, link, post_url, num_post, 1./len(links), post_url_full, link_full]
                        df_rows.append(row)
        df = pd.DataFrame(df_rows, columns=['Date', 'Hyperlink', 'Blog', 'PostNb', 'WeightOfLink', 'PostURL', 'HyperlinkURL'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.to_csv("df_"+filename[16:23]+".csv", index=False)


if __name__ == "__main__":
    names = glob.glob("raw_data/quotes*")
    pool = Pool(processes=len(names))
    pool.map(worker, names)
