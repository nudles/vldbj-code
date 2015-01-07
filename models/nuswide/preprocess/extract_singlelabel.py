import os
import sys

fd=open(sys.argv[1])
idx=[]
for line in fd:
    fields=line.split('#$$#')
    labels=fields[0].split(' ')[2:]
    if len(labels)==1:
        idx.append(fields[0].split(' ')[0])
print len(idx)
fd.close()

with open(sys.argv[2], 'w') as fd:
    for x in idx:
        fd.write('%s\n' % x)
    fd.flush()
    fd.close()
print 'Finish'

