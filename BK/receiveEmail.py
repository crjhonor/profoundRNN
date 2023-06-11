from email.parser import Parser
import email.contentmanager as CT
import poplib

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
        print(self.server.getwelcome().decode(('utf-8')))

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
            print(part.get_content_type(),
                  part.get_content_maintype() == 'multipart',
                  part.is_multipart())
        # # Need to process the Plain text and HTML text. Using dictionary to store the text and necessary information.
        # Email Type has to be Checked, if there is no key 'Delivered-To', then I should withdraw.
        if partList[0]['TO'] == 'jaychen175@163.com':
            # Data structure.
            # ...textDict, dictionary to store content received by a single email.
            textDict = dict()
            textDict['Date'] = partList[0]['Date']
            textDict['Subject'] = partList[0]['Subject']
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
        for i in range(len(mails)):
            msg = self.retrieveMsg(i+1)
            textDict = dict()
            textDict = self.contentDecode(msg)
            if textDict == None:
                continue
            else:
                allEmailsList.append(textDict)
        print('Receiving emails DONE!..........')
        return allEmailsList


# Connecting to the email server.=======================================================================================
pop3_server = 'pop.zoho.com.cn'
username = 'chenrujie2022'
password = 'Jay275699.'

receiveEmailInstance = receiveEmail(pop3_server, username, password)
allEmailsList = receiveEmailInstance.allEmails()
# print("Done!")