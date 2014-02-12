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

import sys, os, re, random, math, urllib2, time, cPickle
import numpy

import onlineldavb

def main():
    """
    Displays topics fit by onlineldavb.py. The first column gives the
    (expected) most prominent words in the topics, the second column
    gives their (expected) relative prominence.
    """
    vocab = str.split(file(sys.argv[1]).read())
    testlambda = numpy.loadtxt(sys.argv[2])

    #testlambda2 = testlambda
    #for w in range(0, len(testlambda[0,:])):
    #    testlambda2[:,w] = testlambda2[:,w]/sum(testlambda2[:,w])

    for k in range(0, len(testlambda)):
        
        lambdak = list(testlambda[k, :])
        lambdak = lambdak / sum(lambdak)
        
        #lambdak2 = list(testlambda2[k, :])
        #lambdak2 = lambdak2 / sum(lambdak2)
        
        #lambdak3 = lambdak * lambdak2
        #lambdak3 = lambdak3 / sum(lambdak3)        
        
        temp = zip(lambdak, range(0, len(lambdak)))
        temp = sorted(temp, key = lambda x: x[0], reverse=True)
        print 'topic %d:' % (k)
        # feel free to change the "53" here to whatever fits your screen nicely.
        nToPrint = 10
        if len(temp)<nToPrint:
            nToPrint = len(temp)
        for i in range(0, nToPrint):
            if temp[i][0] > 0.002:
                print '%20s  \t---\t  %.4f' % (vocab[temp[i][1]], temp[i][0])
        print

if __name__ == '__main__':
    main()
