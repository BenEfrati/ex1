
## GMail API

run gmail.py to gain permission for gmail api


```python
#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import base64

import httplib2
import os
import sys

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

try:
    import argparse
    flags = \
        argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/gmail-python-quickstart.json

SCOPES = 'https://www.googleapis.com/auth/gmail.readonly'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Gmail API Python Quickstart'


def get_credentials():
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'gmail-python-quickstart.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE,
                SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else:

              # Needed only for compatibility with Python 2.6

            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials


def main():
    """Shows basic usage of the Gmail API.

    Creates a Gmail API service object and outputs a list of label names
    of the user's Gmail account.
    """

    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('gmail', 'v1', http=http)

    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])

    if not labels:
        print('No labels found.')
    else:
        print('Labels:')
        for label in labels:
            print(label['name'])
    
```

### Search GMail
1. Get email from current userId and save to CSV file
2. Get list of user's emails
3. Get user's emails content
4. Data preprocessing - remove new lines, tabs, numbers and save only letters
5. Then, we found the top 5 email senders and filter the emails
6. Translate messages to english



```python
from apiclient import errors
import csv
import re
import email
import sys
import collections
from googletrans import Translator
import string

def ListMessagesMatchingQuery(service, user_id, query=''):
    try:
        response = service.users().messages().list(userId=user_id,
                q=query).execute()
        messages = []
        if 'messages' in response:
            messages.extend(response['messages'])

        while 'nextPageToken' in response:
            page_token = response['nextPageToken']
            response = service.users().messages().list(userId=user_id,
                    q=query, pageToken=page_token).execute()
            messages.extend(response['messages'])

        return messages
    except errors.HttpError, error:
        print('An error occurred: %s' % error)


def ListMessagesWithLabels(service, user_id, label_ids=[]):
    try:
        response = service.users().messages().list(userId=user_id,
                labelIds=label_ids).execute()
        messages = []
        if 'messages' in response:
            messages.extend(response['messages'])

        while 'nextPageToken' in response:
            page_token = response['nextPageToken']
            response = service.users().messages().list(userId=user_id,
                    labelIds=label_ids, pageToken=page_token).execute()
            messages.extend(response['messages'])

        return messages
    except errors.HttpError, error:
        print('An error occurred: %s' % error)


def saveListToCsv(mydict, csvName):
    with open(csvName + '.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for (key, value) in mydict.items():
            writer.writerow([key, value])


def getEmails(filename):
    print('getting emails')
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('gmail', 'v1', http=http)
    list = ListMessagesMatchingQuery(service, 'me', query='')
    print(str(len(list)))
    with open(filename + '.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for index in range(len(list)):
            writer.writerow([list[index]['id'], list[index]['threadId'
                            ]])
    print('done')


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ''


def getEmailsMsgs(emailsListFileName, emailsBodyFileName):

    # reading the emails list with the ids:

    with open(emailsListFileName + '.csv', 'rb') as csv_file:
        reader = csv.reader(csv_file)
        mydict = dict(reader)
    print('done reading the emails list')
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('gmail', 'v1', http=http)
    
    i = 0
    result = []
    for msgId in mydict:
        i = i + 1
        message = service.users().messages().get(userId='me', id=msgId,
                format='raw').execute()
        msg_str = base64.urlsafe_b64decode(message['raw'].encode('utf-8'
                ))
        mime_msg = email.message_from_string(msg_str)
        fromMsg = find_between(mime_msg['from'], '<', '>').strip()
        if fromMsg == '':
            fromMsg = mime_msg['from']
        bodyMsg = rec_get_payload(mime_msg).replace('\r', ' '
                ).replace('\n', ' ').replace('\t', ' ').replace(',', ' '
                ).replace('.', ' ').replace('"', ' ').replace('/', ' ')
        try:
            bodyMsg = bodyMsg.decode('Windows-1255').encode('utf-8')
        except UnicodeDecodeError:
            pass
        if '<html>' in bodyMsg:
            bodyMsg = bodyMsg[:bodyMsg.index('<html>')]
        result.append([fromMsg, bodyMsg])
        print('msg num: ' + str(i))

        with open('test.csv', 'wb') as myfile:
            wr = csv.writer(myfile)
            wr.writerows(result)
    print('done decoding')


def rec_get_payload(mime_msg):
    ans = ''
    if mime_msg.is_multipart():
        for p in mime_msg.get_payload():
            if p.is_multipart():
                ans = ans + rec_get_payload(p)
            else:
                ans = ans + p.get_payload()
    else:
        ans = ans + mime_msg.get_payload()
    return ans


def preapare_data(bodyOfEmailsFileName, clearFileName):    
    csv.field_size_limit(sys.maxint)
    with open(bodyOfEmailsFileName + '.csv', 'rb') as csv_file:
        reader = csv.reader(csv_file)
        with open(clearFileName + '.csv', 'wb') as csv_file2:
            line = next(reader, None)
            i = 0
            while line:
                writer = csv.writer(csv_file2)
                print(i)
                i = i + 1
                editedLine = line[1]
                for ch in ['&', '#','-','!','@','$','%','^','&','*','(',')','_','+','=','/','\\','?',':',';','~','1','2','3','4','5','6','7','8','9','>','<','|','{','}','[',']']:
                    editedLine = editedLine.replace(ch, ' ')                
                editedLine = re.sub(' +', ' ', editedLine)
                writer.writerow([line[0].strip(), editedLine])
                line = next(reader, None)


def getTopSendsers(sourceFileName):    
    csv.field_size_limit(sys.maxint)
    with open(sourceFileName + '.csv', 'rb') as csv_file:
        reader = csv.reader(csv_file)
        line = next(reader, None)
        i = 0
        senders = []
        while line:
            print(i)
            i = i + 1
            sender = line[0]
            senders.append(sender)
            line = next(reader, None)        
        counter = collections.Counter(senders)
        print(counter)


def filterBySenders(srcFileName, destinationFileName, senderlist):    
    csv.field_size_limit(sys.maxint)
    with open(srcFileName + '.csv', 'rb') as csv_file:
        reader = csv.reader(csv_file)
        with open(destinationFileName + '.csv', 'wb') as csv_file2:
            line = next(reader, None)
            i = 0
            while line:
                writer = csv.writer(csv_file2)
                print(i)
                i = i + 1
                if line[0] in senderlist:
                    writer.writerow([line[0].strip(), line[1]])
                line = next(reader, None)


def translateToEnglish(srcFilePath, destinationFileName, start, end):    
    with open(srcFilePath + '.csv', 'rb') as csv_file:
        reader = csv.reader(csv_file)
        with open(destinationFileName + '.csv', 'ab') as csv_file2:
            line = next(reader, None)
            i = 0
            while line:
                writer = csv.writer(csv_file2)
                print(i)
                i = i + 1
                line_text = (line[1])[0:4999]
                try:
                    if i > start and i <= end:
                        translator = Translator()
                        translated = translator.translate(line_text)                        
                        printable = set(string.printable)
                        translated_text = filter(lambda x: x \
                                in printable, translated.text)
                        writer.writerow([line[0], translated_text])
                    line = next(reader, None)
                    if i > end:
                        break
                except:
                    print('skipped: ' + str(i))


if __name__ == '__main__':

    # main()
    # getEmails('emailsListIDS')
    # getEmailsMsgs('emailsListIDS','emailsBodyList')
    # preapare_data('test','clearEmailsBodyList2')
    # getTopSendsers('clearEmailsBodyList2')
    # #I choose the 5 senders email addresses I want to extract from the whole DB
    # senderlist=['dean@bgu.ac.il','peler@exchange.bgu.ac.il','bitahon@bgu.ac.il','career@bgu.ac.il','shanigu@bgu.ac.il']
    # filterBySenders('clearEmailsBodyList2','filteredBySenders',senderlist)
    # translateToEnglish('filteredBySenders','filteredBySendersTranslated',821,880)

    print('done.')
```
