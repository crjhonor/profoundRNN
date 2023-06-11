"""
Purpose of version 2 compare with previous version developed before is to achieve more accurate prediction results of
day +1 and day +2 change.
"""

"""
Version 2, code structures:
Codes are going to be given suffix "_V2" in version 2 codes, in order to distinguish them from version one even thought
I place them into the same directory.

    <dicinger_V2.py> Code structure of the dincinger_V2.py has been well organized for better reading. And, there are
    three features for deep learning. First is the same as sentiment derived from emails; second is the sentiment 
    derived from weibo; third is still the log return of yields. Now features are in the shape of [..., 3, 4] which 
    should be more ideal for deep learning.
    
    <getWeibo_V2.py> With the ability to generate and return weibo features to the dicinger_V2.py code, this code gets 
    data from given uids' weibo and storing them into mariaDB databases.
    
    <weiboSentimentcode_V2.py> is the sentiment code which imports getWeibo_V2.py to crawl weibo and store contents into
    database and then, perform bilstm module sentiment on them. Total sentiment results are then sent to
    weiboSentianalysis_V2.py code to generate desired weibo sentiment features.
    
    <weiboSentianalysis_V2.py> is the code to transform the sentiment results into desired weibo sentiment feeatures.
    
    <getYields_V2.py> This code directly read from yields.xls stored in the data directory and it also calculates the 
    log return on daily bases. 
    
    
"""

"""
mariaDB database is used to store the Weibo data so far. Using user root as the username and password is "275699".
"""

"""
Version 2.1
UPDATE:
# # TODO: 
"""

# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Versison 2.0, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('of sentiCourse project.')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
