# # TODO: in receiveEmail_V2, I am going to store the contents from emails I retrieved into database.

from email.parser import Parser
import email.contentmanager as CT
import poplib
from tqdm import tqdm
import pymysql
import pandas as pd
import datetime, pytz
pt = pytz.timezone('Asia/Shanghai')

class receiveEmail:
    '''object oriented programming the email receive process.'''
    def __init__(self, pop3_server, username, password):
        self.pop3_server = pop3_server
        self.username = username
        self.password = password

    def connecServer(self):
        self.server = poplib.POP3_SSL(self.pop3_server)
        self.server.user(self.username)
        self.server.pass_(self.password)
        print('\n', self.server.getwelcome().decode(('utf-8')))

    def retrieveMsg(self, mailNum = 1):
        server = self.server
        resp, mailContent, octest = server.retr(mailNum)
        toParse = b'\r\n'.join(mailContent).decode('utf-8')
        msg = Parser().parsestr(toParse)
        return msg

    def contentDecode(self, msg):
        # # Getting information of the email.
        partList = list()
        for part in msg.walk():
            partList.append(part)
            # print(part.get_content_type(),
            #       part.get_content_maintype() == 'multipart',
            #       part.is_multipart())
        # # Need to process the Plain text and HTML text. Using dictionary to store the text and necessary information.
        # Email Type has to be Checked, if there is no key 'Delivered-To', then I should withdraw.
        if partList[0]['TO'] == 'jaychen175@163.com':
            # Data structure.
            # ...textDict, dictionary to store content received by a single email.
            textDict = dict()
            textDict['EMAILDATE'] = partList[0]['Date']
            textDict['SUBJECT'] = partList[0]['Subject']
            textDict['MID'] = "".join([str(ord(i)) for i in partList[0]['Date']])
            for j in range(len(partList)-1):
                if partList[j+1].get_content_type() == 'text/plain':
                    textDict['PLAINTEXT'] = CT.get_text_content(partList[j+1])
                elif partList[j+1].get_content_type() == 'text/html':
                    textDict['HTMLTEXT'] = CT.get_text_content(partList[j+1])
            return textDict
        elif partList[0]['TO'] == 'chenrujie2022@zoho.com.cn':
            # Data structure.
            # ...textDict, dictionary to store content received by a single email.
            textDict = dict()
            textDict['EMAILDATE'] = partList[0]['Date']
            textDict['SUBJECT'] = partList[0]['Subject']
            textDict['MID'] = "".join([str(ord(i)) for i in partList[0]['Date']])
            for j in range(len(partList) - 1):
                if partList[j + 1].get_content_type() == 'text/plain':
                    textDict['PLAINTEXT'] = CT.get_text_content(partList[j + 1])
                elif partList[j + 1].get_content_type() == 'text/html':
                    textDict['HTMLTEXT'] = CT.get_text_content(partList[j + 1])
            return textDict
        else:
            return None

    def allEmails(self, dateBacktime=1): # Main functions to get returning all decoding email messages in list objects.
        # connect to the database to see the latest storage email
        def _getTodate():
            conn = pymysql.connect(host='127.0.0.1', user='root', passwd="275699", db='email')
            databaseRead = []
            try:
                with conn.cursor() as cursor:
                    sql = "SELECT * FROM email2_tbl ORDER BY EMAILDATE DESC"
                    cursor.execute(sql)
                    for read in cursor:
                        readDict = {
                            "EMAILDATE": read[0],
                            'SUBJECT': read[1],
                            'MID': read[2],
                            'PLAINTEXT': read[3],
                            'HTMLTEXT': read[4]
                        }
                        databaseRead.append(readDict)
                conn.commit()
            finally:
                conn.close()
            databaseRead_df = pd.DataFrame(databaseRead,
                                           columns=["EMAILDATE", "SUBJECT", "MID", "PLAINTEXT", "HTMLTEXT"])
            time_line = pd.to_datetime(databaseRead_df['EMAILDATE'])
            latest_time = time_line.max()
            return databaseRead, latest_time

        databaseRead, latest_time = _getTodate()
        # Read the system time and subtract it by 1 month, then we should start reading email from server from that
        # time. In order not to miss any unread email, I should check whether the latest_time on the database is earlier
        # than the system sub 1 month
        sysTime = datetime.datetime.now()
        readFromtime = sysTime - datetime.timedelta(days=dateBacktime*30)
        if readFromtime.astimezone(pt) >= latest_time.astimezone(pt):
            readFromtime = latest_time

        # Read emails
        self.connecServer()
        retrieveEmailsList = list()
        server = self.server
        resp, mails, octest = server.list()
        for i in tqdm(range(len(mails)), ncols=150, desc="Receiving emails"):
            j = len(mails) - i # I have to fetch emails from the latest one
            msg = self.retrieveMsg(j)
            textDict = dict()
            textDict = self.contentDecode(msg)
            if textDict == None:
                continue
            elif readFromtime.astimezone(pt) >= pd.to_datetime(textDict['EMAILDATE']).astimezone(pt):
                print("\n", "Receiving Emails Done!", "." * 200, "\n")
                # Combine the retrieveEmailsList and databaseRead_df to construct the allEmailsList for return
                for item in retrieveEmailsList:
                    databaseRead.append(item)
                return databaseRead, retrieveEmailsList
            else:
                retrieveEmailsList.append(textDict)

# Use mariaDB to store the email data.==================================================================================
class emailDBstorage:
    def __init__(self, emailData):
        self.emailData = emailData
        self.storeTodatabase()

    def emailInsert(self, emaildate, subject, mid, plaintext, htmltext):
        conn = pymysql.connect(host='127.0.0.1', user='root', passwd="275699", db='email')
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO email2_tbl(EMAILDATE, SUBJECT, MID, PLAINTEXT, HTMLTEXT) VALUES (%s, %s, %s, %s, %s)"
                try:
                    cursor.execute(sql, (emaildate, subject, mid, plaintext, htmltext))
                except pymysql.err.IntegrityError:
                    conn.commit()
                finally:
                    conn.commit()
        finally:
            conn.close()

    def storeTodatabase(self):
        emailData = self.emailData
        for i in tqdm(range(len(emailData)), ncols=150, desc="Store emails into database"):
            insertData = emailData[i]
            try:
                emaildate =insertData['EMAILDATE']
                subject = insertData['SUBJECT']
                mid = insertData['MID']
                plaintext = insertData['PLAINTEXT']
                htmltext = insertData['HTMLTEXT']
                self.emailInsert(emaildate, subject, mid, plaintext, htmltext)
            except KeyError:
                continue
        print("\n", "Storing emails to database DONE!", "."*200, "\n")

# Connecting to the email server.=======================================================================================
pop3_server = 'pop.zoho.com.cn'
username = 'chenrujie2022'
password = 'Jay275699.'
dateBacktime = 1 # number of months date back to when start the receiving of emails.

receiveEmailInstance = receiveEmail(pop3_server, username, password)
allEmailsList, retrieveEmailsList = receiveEmailInstance.allEmails(dateBacktime=dateBacktime)
# Check if the database is ready
# tal, tre = [], []
# for item in allEmailsList:
#     try:
#         tal.append(item[0])
#     except KeyError:
#         continue
# for item in retrieveEmailsList:
#     try:
#         tre.append(item['Date'])
#     except KeyError:
#         continue
# tal_dt = pd.to_datetime(tal)
# tre_dt = pd.to_datetime(tre)
# tal_dt.values.sort()
# tre_dt.values.sort()
# tal_dt[-10:]
# tre_dt[-10:]
edb = emailDBstorage(retrieveEmailsList)
# print("Done!")