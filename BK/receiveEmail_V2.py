# # TODO: in receiveEmail_V2, I am going to store the contents from emails I retrieved into database.

from email.parser import Parser
import email.contentmanager as CT
import poplib
from tqdm import tqdm
import pymysql

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
            textDict['Date'] = partList[0]['Date']
            textDict['Subject'] = partList[0]['Subject']
            textDict['MID'] = "".join([str(ord(i)) for i in partList[0]['Date']])
            for j in range(len(partList)-1):
                if partList[j+1].get_content_type() == 'text/plain':
                    textDict['plainText'] = CT.get_text_content(partList[j+1])
                elif partList[j+1].get_content_type() == 'text/html':
                    textDict['htmlText'] = CT.get_text_content(partList[j+1])
            return textDict
        elif partList[0]['TO'] == 'chenrujie2022@zoho.com.cn':
            # Data structure.
            # ...textDict, dictionary to store content received by a single email.
            textDict = dict()
            textDict['Date'] = partList[0]['Date']
            textDict['Subject'] = partList[0]['Subject']
            textDict['MID'] = "".join([str(ord(i)) for i in partList[0]['Date']])
            for j in range(len(partList) - 1):
                if partList[j + 1].get_content_type() == 'text/plain':
                    textDict['plainText'] = CT.get_text_content(partList[j + 1])
                elif partList[j + 1].get_content_type() == 'text/html':
                    textDict['htmlText'] = CT.get_text_content(partList[j + 1])
            return textDict
        else:
            return None

    def allEmails(self): # Main functions to get returning all decoding email messages in list objects.
        self.connecServer()
        allEmailsList = list()
        server = self.server
        resp, mails, octest = server.list()
        for i in tqdm(range(len(mails)), ncols=150, desc="Receiving emails"):
            msg = self.retrieveMsg(i+1)
            textDict = dict()
            textDict = self.contentDecode(msg)
            if textDict == None:
                continue
            else:
                allEmailsList.append(textDict)
        print("\n", "Receiving Emails Done!", "." * 200, "\n")
        return allEmailsList

# Use mariaDB to store the email data.==================================================================================
class emailDBstorage:
    def __init__(self, emailData):
        self.emailData = emailData
        self.storeTodatabase()

    def emailInsert(self, emaildate, subject, mid, plaintext, htmltext):
        conn = pymysql.connect(host='127.0.0.1', user='root', passwd="275699", db='email')
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO email_tbl(EMAILDATE, SUBJECT, MID, PLAINTEXT, HTMLTEXT) VALUES (%s, %s, %s, %s, %s)"
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
                emaildate =insertData['Date']
                subject = insertData['Subject']
                mid = insertData['MID']
                plaintext = insertData['plainText']
                htmltext = insertData['htmlText']
                self.emailInsert(emaildate, subject, mid, plaintext, htmltext)
            except KeyError:
                continue
        print("\n", "Storing emails to database DONE!", "."*200, "\n")

# Connecting to the email server.=======================================================================================
pop3_server = 'pop.zoho.com.cn'
username = 'chenrujie2022'
password = 'Jay275699.'

receiveEmailInstance = receiveEmail(pop3_server, username, password)
allEmailsList = receiveEmailInstance.allEmails()
edb = emailDBstorage(allEmailsList)
# print("Done!")