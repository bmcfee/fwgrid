#!/usr/bin/env python

import scipy.io
import numpy
from pprint import pprint
import sys

title = sys.argv[1]
infile = sys.argv[2]

P = scipy.io.loadmat(infile,struct_as_record=False)

scores = P['scores']
mu = numpy.mean(scores)
sig = numpy.std(scores)
print '"%s",%.3f,%.3f' % (title, mu, sig)
# print '"%s",%s' % (title, ','.join(['%f' % (1-x) for x in P['scores']]))
# print '"%s",%s' % (title, ','.join(['%f' % x for x in P['scores']]))
