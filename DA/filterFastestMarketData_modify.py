import sys

inputFileName = sys.argv[1]
outputFileName = sys.argv[2]

if (len(sys.argv) >= 4):
    sourceList = sys.argv[3].split(';')
else:
    sourceList = []

inputFile = open(inputFileName, 'r')

inputLines = []
while True:
    line = inputFile.readline()
    if line == '':
        break
    spt = line.strip().split(',')
    try:
        hour = int(spt[1][9:11])
        if hour > 15:
            continue
    except:
        continue
    line = ','.join(spt) + '\n'
    inputLines.append(line)


inputFile.close()
outputFile = open(outputFileName, 'w')
#before20190401
# pp = ['pp2002,20181102 18:34:03.000,20181101 20:45:04.724817933,NaN,0,NaN,0,NaN,0,NaN,0,NaN,0,NaN,0,NaN,0,NaN,0,NaN,0,NaN,0,NaN,NaN,9296.000000,10274.000000,NaN,419670,NaN,9742.000000,419670,9785.000000,0,0,XSPEED_L2_TCP,1\n','l2002,20181102 18:34:03.000,20181101 20:45:04.724817933,NaN,0,NaN,0,NaN,0,NaN,0,NaN,0,NaN,0,NaN,0,NaN,0,NaN,0,NaN,0,NaN,NaN,9296.000000,10274.000000,NaN,419670,NaN,9742.000000,419670,9785.000000,0,0,XSPEED_L2_TCP,1\n']
# if outputFileName.find('XSpeed_L1_L2') >= 0:
    # outputFile.writelines(pp+outputLines)
# else:
    # outputFile.writelines(outputLines)
outputFile.writelines(inputLines)
outputFile.close()
