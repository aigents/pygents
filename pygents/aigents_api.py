# MIT License
# 
# Copyright (c) 2015-2023 Aigents®, Anton Kolonin 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Service wrapper around Aigents Java-based Web Service
"""        

import requests
import urllib.parse
import math
import json


import logging
logger = logging.getLogger(__name__)	

class AigentsAPI:

	def __init__(self, base_url='https://aigents.com/al', login_email='john@doe.org', secret_question='password', secret_answer='1234567', real_mode=True, name='Aigents', verbose=False):
		self.name = name
		self.verbose = verbose
		self.base_url = base_url # Aigents Web API hosting URL
		self.login_email = login_email # Aigents user identification by email 
		self.secret_question = secret_question # Aigents prompt for password
		self.secret_answer = secret_answer # Aigents password value
		self.real_mode = real_mode # whether to connect to real Aigents server (True) or fake test oe (False) 
		if self.verbose:
			logger.info('Creating Aigents session')
		#print(self)
		self.create_session()
	
	def __del__(self):
		if self.verbose:
			logger.info('Closing Aigents session')
		self.close_session()

	def create_session(self):
		self.session = requests.session()
		if self.real_mode:
			#TODO assertions
			self.request('my email ' + self.login_email + '.')
			self.request('my ' + self.secret_question + ' '  + self.secret_answer + '.')
			self.request('my language english.')
		else:
			#TODO make sure if we can use only one of these
			output = self.request('my name ' + self.login_email + ', surname ' + self.login_email + ', email ' + self.login_email + '.')
			if output == 'What your secret question, secret answer?':
				assert output == 'What your secret question, secret answer?', 'Expecting secret question, secret answer'
				output = self.request('my secret question ' + self.secret_question + ', secret answer ' + self.secret_answer + '.')
			assert output == 'What your ' + self.secret_question + '?', 'Expecting secret question'
			output = self.request('my ' + self.secret_question + ' ' + self.secret_answer + '.')
			assert output.split()[0] == 'Ok.', 'Expecting Ok'

	def close_session(self):
		if not self.real_mode:
			output = self.request('Your trusts no ' + self.login_email + '.')
			assert output.split()[0] == 'Ok.', 'Expecting Ok'
			output = self.request('No name ' + self.login_email + '.');
			assert output.split()[0] == 'Ok.', 'Expecting Ok'
			output = self.request('No there times today.');
			assert output.split()[0] == 'Ok.', 'Expecting Ok'
		output = self.request('My logout.');
		assert output.split()[0] == 'Ok.', 'Expecting Ok'
			
	def request(self,input):
		if self.verbose:
			logger.info(input)
		url = self.base_url + '?' + urllib.parse.quote_plus(input)
		try:
			r = self.session.post(url)
			if r is None or r.status_code != 200:
				logger.error('request ' + url + ' error ' + str(r.status_code))
				raise RuntimeError("Aigents - no response")
		except Exception as e:
			logger.error('request ' + url + ' ' + str(type(e)))
			print('Specify proper url to Aigents server or run it locally, eg.')
			print('java -cp ./bin/Aigents.jar:./bin/* net.webstructor.agent.Farm store path \'./al_test.txt\', cookie domain localtest.com, console off &')
			print('or')
			print('sh aigents_server_start.sh')
			return 'No connection to Aigents, ' + str(type(e))
		if self.verbose:
			logger.info(r.text)
		return r.text
	
	
class AigentsSentiment():

    def __init__(self, api_url='http://localtest.com:1180/', debug=False):
        self.aa = AigentsAPI(api_url)
        self.debug = debug
        reply = self.aa.request("my format json")
        if debug:
            print(reply)
            
    def get_sentiment(self,text,context=None,debug=False):
        
        wordcnt = len(text.strip().split(' '))
        itemcnt = 1
    
        text_quoted = '"'+text.replace('"','\\"')+'"'
        request_text = "classify sentiment text "+text_quoted
        json_text = self.aa.request(request_text)
        if self.debug:
            print('--------')
            print('text:',text) 
            print('json:',json_text)
            
        sen = 0
        pos = 0
        neg = 0
        con = 0
            
        try:
            python_data = json.loads(json_text)
            item = python_data[0]
            #print(text, item['sentiment'])
            sen = float(item['sentiment']) / 100
            pos = float(item['positive']) / 100 if 'positive' in item.keys() else 0
            neg = -float(item['negative']) / 100 if 'negative' in item.keys() else 0
            con = round(math.sqrt(pos * -neg),2)
        except Exception as e:
            print(str(e), request_text,'=>',json_text)
            if self.debug:
                raise e
        return sen, pos, neg, con, wordcnt, itemcnt
	
