#!/usr/bin/python2.7

import sys
import os
import random

if __name__=='__main__':
	domainName = sys.argv[1]

	if os.path.exists("./" + domainName + "/dev/pre_trad_me_reorder.xml") == False:
		os.rename("./" + domainName + "/dev/trad_me_reorder.xml" ,"./" + domainName + "/dev/pre_trad_me_reorder.xml")
	inFile = open("./" + domainName + "/dev/pre_trad_me_reorder.xml")
	outFile = open("./" + domainName + "/dev/trad_me_reorder.xml", "w")

	mono = []
	invert = []
	for line in inFile:
		if line.find('mono') == 0:
			mono.append(line)
		else:
			invert.append(line)

	for i in range(0,250):
		a = random.randint(0, len(mono)-1)
		while mono[a] == 'null':
			a = random.randint(0, len(mono)-1)

		outFile.write(mono[a])
		mono[a] = 'null'

	for i in range(0,250):
		a = random.randint(0, len(invert)-1)
		while invert[a] == 'null':
			a = random.randint(0, len(invert)-1)

		outFile.write(invert[a])
		invert[a] = 'null'

	inFile.close()
	outFile.close()

	inFile = open("./" + domainName + "/dev/pre_trad_me_reorder.xml");
	outFile = open("./" + domainName + "/dev/train_trad_me_reorder.xml", "w")

	mono = []
	invert = []
	for line in inFile:
		if line.find('mono') == 0:
			mono.append(line)
		else:
			invert.append(line)

	minLen = 0
	if len(mono) < 10000:
		if len(mono) > len(invert):
			minLen = len(invert)-1
		else:
			minLen = len(mono)-1
	elif len(invert) < 10000:
		if len(mono) > len(invert):
			minLen = len(invert)-1
		else:
			minLen = len(mono)-1
	else:
		minLen = 10000

	for i in range(0,minLen):
		a = random.randint(0, len(mono)-1)
		while mono[a] == 'null':
			a = random.randint(0, len(mono)-1)

		outFile.write(mono[a])
		mono[a] = 'null'

	for i in range(0,minLen):
		a = random.randint(0, len(invert)-1)
		while invert[a] == 'null':
			a = random.randint(0, len(invert)-1)

		outFile.write(invert[a])
		invert[a] = 'null'

	inFile.close()
	outFile.close()

	if os.path.exists("./" + domainName + "/test/pre_trad_me_reorder.xml") == False:
		os.rename("./" + domainName + "/test/trad_me_reorder.xml" ,"./" + domainName + "/test/pre_trad_me_reorder.xml")
	inFile = open("./" + domainName + "/test/pre_trad_me_reorder.xml");
	outFile = open("./" + domainName + "/test/trad_me_reorder.xml", "w");

	mono = []
	invert = []
	for line in inFile:
		if line.find('mono') == 0:
			mono.append(line)
		else:
			invert.append(line)

	for i in range(0,250):
		a = random.randint(0, len(mono)-1)
		while mono[a] == 'null':
			a = random.randint(0, len(mono)-1)

		outFile.write(mono[a])
		mono[a] = 'null'

	for i in range(0,250):
		a = random.randint(0, len(invert)-1)
		while invert[a] == 'null':
			a = random.randint(0, len(invert)-1)

		outFile.write(invert[a])
		invert[a] = 'null'

	inFile.close()
	outFile.close()

	if os.path.exists("./" + domainName + "/train/pre_trad_me_reorder.xml") == False:
		os.rename("./" + domainName + "/train/trad_me_reorder.xml" ,"./" + domainName + "/train/pre_trad_me_reorder.xml")
	inFile = open("./" + domainName + "/train/pre_trad_me_reorder.xml");
	outFile = open("./" + domainName + "/train/trad_me_reorder.xml", "w");

	mono = []
	invert = []
	for line in inFile:
		if line.find('mono') == 0:
			mono.append(line)
		else:
			invert.append(line)

	for i in range(0,5000):
		a = random.randint(0, len(mono)-1)
		while mono[a] == 'null':
			a = random.randint(0, len(mono)-1)

		outFile.write(mono[a])
		mono[a] = 'null'

	for i in range(0,5000):
		a = random.randint(0, len(invert)-1)
		while invert[a] == 'null':
			a = random.randint(0, len(invert)-1)

		outFile.write(invert[a])
		invert[a] = 'null'

	inFile.close()
	outFile.close()
