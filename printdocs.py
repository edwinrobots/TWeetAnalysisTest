#!/usr/bin/python

# printtopics.py: Prints the words that are most prominent in a set of
# topics.
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cPickle
import math
import numpy
import os
import random
import re
import sys
import time
import urllib2

import onlineldavb


def main():
    """
    Displays topics of each document fit by onlineldavb.py. The first column gives the
    (expected) most prominent words in the topics, the second column
    gives their (expected) relative prominence.
    """
    testgamma = numpy.loadtxt(sys.argv[1])
    vocab = str.split(file(sys.argv[2]).read())
    testlambda = numpy.loadtxt(sys.argv[3])

    nToPrint = len(testgamma)
    for d in range(1460, 1480):
        gammad = list(testgamma[d, :])
        gammad = gammad / sum(gammad)
        topicsford = zip(gammad, range(0, len(gammad)))
        topicsford = sorted(topicsford, key = lambda x: x[0], reverse=True)
        print 'document %d:' % (d)
        # feel free to change the "53" here to whatever fits your screen nicely.
        for i in range(0, 2):
            k = topicsford[i][1]
            lambdak = list(testlambda[k, :])
            lambdak = lambdak / sum(lambdak)
            temp = zip(lambdak, range(0, len(lambdak)))
            temp = sorted(temp, key = lambda x: x[0], reverse=True)
            
            topicstring = ''
            for j in range(0,10):
                topicstring += ' ' 
                topicstring += vocab[temp[j][1]]
            
            print '%d  \t---\t  %.4f \t---\t %s' % (topicsford[i][1], topicsford[i][0], topicstring)
        print

if __name__ == '__main__':
    main()
