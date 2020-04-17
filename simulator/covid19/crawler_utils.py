import random
import pandas as pd
import requests
import time
import sys


def get_user_agent():
    user_agent_list = [
        'Mozilla/1.22 (compatible; MSIE 10.0; Windows 3.1)',
        'Mozilla/4.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/5.0)',
        'Mozilla/4.0 (Compatible; MSIE 8.0; Windows NT 5.2; Trident/6.0)',
        'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)',
        'Mozilla/5.0 (compatible; MSIE 10.0; Macintosh; Intel Mac OS X 10_7_3; Trident/6.0)',
        'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/4.0; InfoPath.2; SV1; .NET CLR 2.0.50727; WOW64)',
        'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/5.0)',
        'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)',
        'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)',
        'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
        'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
        'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 7.0; InfoPath.3; .NET CLR 3.1.40767; Trident/6.0; en-IN)',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.0; Trident/5.0)',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML like Gecko) Chrome/55.0.2883.87 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 5.1; Win64; x64) AppleWebKit/537.36 (KHTML like Gecko) Chrome/60.0.3112.90 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML like Gecko) Chrome/55.0.2883.87 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML like Gecko) Chrome/57.0.2987.133 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML like Gecko) Chrome/60.0.3112.90 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.2; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML like Gecko) Chrome/60.0.3112.113 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML like Gecko) Chrome/44.0.2403.157 Safari/537.36',
        'Opera/12.02 (Android 4.1; Linux; Opera Mobi/ADR-1111101157; U; en-US) Presto/2.9.201 Version/12.02',
        'Opera/9.80 (Android 2.3.3; Linux; Opera Mobi/ADR-1111101157; U; es-ES) Presto/2.9.201 Version/11.50',
        'Opera/9.80 (S60; SymbOS; Opera Mobi/SYB-1107071606; U; en) Presto/2.8.149 Version/11.10',
        'Opera/9.80 (Android 2.3.4; Linux; Opera Mobi/build-1107180945; U; en-GB) Presto/2.8.149 Version/11.10',
        'Opera/9.80 (Android 2.2.1; Linux; Opera Mobi/ADR-1107051709; U; pl) Presto/2.8.149 Version/11.10',
        'Opera/9.80 (S60; SymbOS; Opera Mobi/SYB-1104061449; U; da) Presto/2.7.81 Version/11.00',
        'Opera/9.80 (S60; SymbOS; Opera Mobi/SYB-1103211396; U; es-LA) Presto/2.7.81 Version/11.00',
        'Opera/9.80 (Android; Linux; Opera Mobi/ADR-1012221546; U; pl) Presto/2.7.60 Version/10.5',
        'Opera/9.80 (Android 2.2;;; Linux; Opera Mobi/ADR-1012291359; U; en) Presto/2.7.60 Version/10.5',
        'Opera/9.80 (Android 2.2; Opera Mobi/ADR-2093533608; U; pl) Presto/2.7.60 Version/10.5',
        'Opera/9.80 (Android 2.2; Opera Mobi/-2118645896; U; pl) Presto/2.7.60 Version/10.5',
        'Opera/9.80 (Android 2.2; Linux; Opera Mobi/ADR-2093533312; U; pl) Presto/2.7.60 Version/10.5',
        'Opera/9.80 (Android 2.2; Linux; Opera Mobi/ADR-2093533120; U; pl) Presto/2.7.60 Version/10.5',
        'Opera/9.80 (Android 2.2; Linux; Opera Mobi/8745; U; en) Presto/2.7.60 Version/10.5',
        'Opera/9.80 (S60; SymbOS; Opera Mobi/1209; U; sk) Presto/2.5.28 Version/10.1',
        'Opera/9.80 (S60; SymbOS; Opera Mobi/1209; U; fr) Presto/2.5.28 Version/10.1',
        'Opera/9.80 (S60; SymbOS; Opera Mobi/1181; U; en-GB) Presto/2.5.28 Version/10.1',
        'Opera/9.80 (Android; Linux; Opera Mobi/ADR-1012211514; U; en) Presto/2.6.35 Version/10.1',
        'Opera/9.80 (Android; Linux; Opera Mobi/ADR-1011151731; U; de) Presto/2.5.28 Version/10.1',
        'Mozilla/5.0 (Windows; U; MSIE 9.0; WIndows NT 9.0; en-US))',
        'Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 7.1; Trident/5.0)',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; chromeframe/12.0.742.112)',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0; yie8)',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0; FunWebProducts)',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0; chromeframe/13.0.782.215)',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0; chromeframe/11.0.696.57)',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0) chromeframe/10.0.648.205',
        'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.0; Trident/5.0; chromeframe/11.0.696.57)',
        'Mozilla/5.0 (X11; Linux i686; rv:64.0) Gecko/20100101 Firefox/64.0',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:64.0) Gecko/20100101 Firefox/64.0',
        'Mozilla/5.0 (X11; Linux i586; rv:63.0) Gecko/20100101 Firefox/63.0',
        'Mozilla/5.0 (Windows NT 6.2; WOW64; rv:63.0) Gecko/20100101 Firefox/63.0',
        'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10.10; rv:62.0) Gecko/20100101 Firefox/62.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:10.0) Gecko/20100101 Firefox/62.0'
    ]

    return random.choice(user_agent_list)


def get_city_beds(codibge):
    retries = 0
    max_retries = 5
    success = False

    # set URL with state and city code
    url = "http://cnes2.datasus.gov.br/Mod_Ind_Tipo_Leito.asp?VEstado=%s&VMun=%s" % (codibge[:2], codibge)

    while not success:
        try:
            # request with headers, avoiding blocking the crawler
            headers = {'User-Agent': get_user_agent()}
            r = requests.get(url, headers=headers)

            # parse data from html to data frame
            df = pd.read_html(r.content, header=0)[3]

            # filter rows with data for beds
            df = df[df.Codigo.str.isdigit()]

            # add codibge
            df.insert(0, 'codibge', codibge)

            success = True

        except IndexError:
            # create blank data frame
            data = [[codibge, '00', 'None', 0, 0, 0]]
            df = pd.DataFrame(data, columns=['codibge', 'Codigo', 'Descrição', 'Existente', 'Sus', 'Não Sus'])
            success = True

        except Exception as e:
            sys.stdout.write('\n Error accessing:%s, %s' % (url, e.__str__()))
            sys.stdout.flush()
            retries += 1
            if retries >= max_retries:
                sys.stdout.write('\n Accessing:%s, error:%s' % (url, e.__str__()))
                sys.stdout.flush()
                data = [[codibge, '00', 'None', 0, 0, 0]]
                df = pd.DataFrame(data, columns=['codibge', 'Codigo', 'Descrição', 'Existente', 'Sus', 'Não Sus'])
                success = True
            else:
                sys.stdout.write('\n Retrying(%s) \n' % retries)
                sys.stdout.flush()
                wait = retries * 30
                time.sleep(wait)

    return df


def get_bed_codes(type):

    if type == 'normal':
        codes_list = ['01', '02', '03', '04', '05', '07', '08', '09', '11', '12', '13', '14', '15', '16', '31', '32',
                         '33', '34', '35', '36', '37', '38', '40', '41', '42', '44', '46', '48', '49', '66', '67', '69',
                         '70', '71', '72', '88', '90', '95']

    if type == 'icu':
        codes_list = ['75', '74', '76', '85', '86', '83']

    if type == 'covid':
        codes_list = ['51']

    return codes_list
