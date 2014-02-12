# -*- coding: utf-8 -*-
#!/usr/bin/python

# projectTweets.py: ANALYSE A SET OF TWEETS AND PROJECT THEM TO A LOWER-DIMENSIONAL FEATURE SPACE USING LDA 
# How much does adjusting the feature projections improve the classification accuracy?
# Is there any theoretical advantage to pairwise comparisons (or is it possibly just an HCI advantage)?
# We should use labels as training for the classification stage and the feature projection (ideally these will not be separate).
# Start with no training labels, only unsupervised features.
# The clustering method must also obtain training labels - do this as a second stage. 
#Once we know the objects are the same/different, we can obtain labels. 
#This has a cost of two interactions, so collapses to being pointless unless the pairwise comparison is useful for HCI, 
#or we can cluster documents at a more fine-grained level in a hierarchy. --> skip the clustering idea for now

# Based on previous work Copyright (C) 2010  Matthew D. Hoffman
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

import numpy, re, random, scipy.special, copy

import onlineldavb
from scipy.spatial.kdtree import KDTree
from sklearn.metrics import roc_curve, auc

def main():    
    nIntsPerIt = 40 # number of interactions per iteration (don't update LDA too often)
    nIts = 10
    renewLda = False
    
    # Load the tweets: the text; the tweet IDs; the categories, used for evaluation 
    (vocab, docset, origDocs, _, catIdxs, catNames) = getSampleTweets()
    catIdxs = numpy.transpose(numpy.matrix(catIdxs))
    nClasses = len(catNames);
    
    #get raw feature matrix
    F = getRawFeatures(docset, vocab);
    
    #get the training data
    initTrIdxs = []
    for j in range(nClasses):
        jIdxs = catIdxs==j
        (jIdxs, _) = numpy.nonzero(jIdxs)
        initTrIdxs += random.sample(jIdxs.tolist()[0], 3)
    
    initTrData = F[initTrIdxs]#[F[i] for i in trIdxs]
    
    #get the test data
    initTestIdxs = range(len(docset))
    initTestIdxs = [t for t in initTestIdxs if t not in initTrIdxs]
    testData = F[initTestIdxs]#[F[i] for i in testIdxs]
    testLabels = catIdxs[initTestIdxs]#[catIdxs[i] for i in testIdxs]

    # get the LDA features. 
    #Lambda contains pseudocounts for the words in topics (rows=topics,columns=words in vocab)
    #Gamma is pseudocounts for the document topics (rows=docs,columns=topics)
    (_, initGamma, initLda) = runUnsupervisedLda(docset, vocab, 50, renewLda, nClasses)
    
    #now run each of the different types

    interactTypes =  ['catRaw', 'catLdaPush', 'categories', 'catLdaTopics']#, 'clusters']
    
    aucs_knn = numpy.zeros((len(interactTypes), nIts+1, nClasses))
    aucs_nb = numpy.zeros((len(interactTypes), nIts+1, nClasses))
        
    for t in range(len(interactTypes)): 
        interactType = interactTypes[t]
        
        trIdxs = initTrIdxs
        trData = initTrData
        trLabels = catIdxs[trIdxs]
        testIdxs = initTestIdxs
        gamma = copy.deepcopy(initGamma)
        lda = copy.deepcopy(initLda)
          
        for i in range(nIts+1): #add one so that we evaluate the final interactions
            #get current data for classifiers
            
            testLabels = catIdxs[testIdxs]
            
            if interactType=='catRaw':
                trData = F[trIdxs]
                testData = F[testIdxs]
            else:
                trData = gamma[trIdxs,:]
                testData = gamma[testIdxs,:]
                #run the knn classifier using the lda components
                (knnLabels, kdtree) = knnClassify(trData, trLabels, testData, nClasses)
                aucs_knn[t,i,:] = printAucs(nClasses, knnLabels, testLabels, 'KNN ' + interactType + ' ' + str(i))
            
            #run the nb classifier using the lda components
            post = nbClassify(trData, trLabels, testData, nClasses)
            aucs_nb[t,i,:] = printAucs(nClasses, post, testLabels, 'NB ' + interactType + ' ' + str(i))
            
            if i==nIts: #if we have evaluated nIts interactions, don't need any more input
                break
            
            #do something interactive, update gammas, then repeat
            #Ask the user for a pairwise comparison
    #         resp = raw_input('Get more info?')
    #         if resp.lower()=='n':
    #             getMoreInput = False
    #             break
            if interactType=='clusters': #improve the features
                (gamma, lda) = interactClusters(nIntsPerIt, gamma, kdtree, origDocs, docset, lda, catIdxs)
                newTrLabels = []
                newTrIdxs = []
            elif interactType=='categories' or interactType=='catRaw':#increase training set
                (newTrLabels, newTrIdxs) = interactCategories(nIntsPerIt, post, catNames, origDocs, catIdxs)           
            elif interactType=='catLdaPush': #get categorical labels and use them to construct pushes
                (gamma, newTrLabels, newTrIdxs) = interactCatLdaPush(nIntsPerIt, post, catNames, origDocs, docset, catIdxs, lda, gamma)
            elif interactType=='catLdaTopics': #get categorical labels and treat them as known topics of the docs
                (gamma, newTrLabels, newTrIdxs) = interactCatLdaTop(nIntsPerIt, post, catNames, origDocs, docset, catIdxs, lda, gamma)
                
            testIdxs = [x for x in testIdxs if x not in newTrIdxs]
            testLabels = catIdxs[testIdxs]
            trIdxs = trIdxs + newTrIdxs 
            #take the values returned by the "interaction" as these are the labels from the "agents"            
            trLabels = numpy.append(trLabels, numpy.matrix(newTrLabels).transpose(), 0)         

    import cPickle
    filename_aucnb = '/home/edwin/git/TWeetAnalysisTest/data/aucs_nb.dat'
    filename_aucknn = '/home/edwin/git/TWeetAnalysisTest/data/aucs_knn.dat'
    with open(filename_aucnb, 'wb') as output:
        cPickle.dump(aucs_nb, output)
    with open(filename_aucknn, 'wb') as output:
        cPickle.dump(aucs_knn, output)           

def topicEntropy(gamma):
    (_, K) = gamma.shape
    sumgamma = numpy.sum(gamma,1)
    betaln = numpy.sum(scipy.special.gammaln(gamma), 1) - scipy.special.gammaln(sumgamma)
    dig_sum = scipy.special.psi(sumgamma)
    dig_k = scipy.special.psi(gamma)
    H = betaln + numpy.multiply((sumgamma - K), dig_sum) - numpy.sum(numpy.multiply((gamma-1),dig_k),1)
    return H
                  
def interactClusters(nInts, gamma, kdtree, origDocs, docset, lda, catIdxs):      
    #Select objects to query. Should have high entropy over theta, i.e. uncertainty over the topic 
#     distribution of the documents, NOT uncertainty over the topics themselves.
  
    #find most uncertain object
    # log Beta(gamma) + (sumgamma - K)psi(sumgamma) - sum_K (gamma_k-1)psi(gamma_k)

    H = topicEntropy(gamma)

    for i in range(nInts):
        
        chosenDoc = numpy.argmax(H)
        
        H[chosenDoc] = float('-inf') # set to zero so next interaction does not use same doc
        
        #find its current nearest neighbour
        (_, idx) = kdtree.query(gamma[chosenDoc,:], 1)
        
        while lda.pushes[chosenDoc, idx] != 0:
            H[chosenDoc] = float('-inf')
            chosenDoc = numpy.argmax(H)
            (_, idx) = kdtree.query(gamma[chosenDoc,:], 1)
         
        #print 'Tweets to compare: ' + str(chosenDoc) + ', ' + str(idx)   
        #print 'Tweet 1: ' + origDocs[chosenDoc]
        #print 'Tweet 2: ' + origDocs[idx]    
#         text = 'Do these two tweets discuss the same specific need, problem or emergency, or are they both irrelevant to rescue/aid workers? Enter Y or N'
        
        if catIdxs[chosenDoc]==catIdxs[idx]:
            label = 1
        else:
            label = -1
#uncomment the stuff below to get real answers from the command line 
#         label=0
#         while label==0:        
#             resp = raw_input(text)
#             if resp.lower()=='y':
#                 label = 1
#             elif resp.lower()=='n':
#                 label = -1
#             else:
#                 print 'Need to enter Y or N'        
                
        lda.insert_push(chosenDoc, idx, label)
    
    (_, gamma, lda) = updateLda(lda, docset, gamma)
    return (gamma, lda)
         
def interactClustersHacky(gamma,kdtree,origDocs,lda):    
    
    (_, K) = gamma.shape
    sumgamma = numpy.sum(gamma,1)
    betaln = numpy.sum(scipy.special.gammaln(gamma), 1) - scipy.special.gammaln(sumgamma)
    dig_sum = scipy.special.psi(sumgamma)
    dig_k = scipy.special.psi(gamma)
    H = betaln + numpy.multiply((sumgamma - K), dig_sum) - numpy.sum(numpy.multiply((gamma-1),dig_k),1)
    chosenDoc = numpy.argmax(H)
    
    #find its current nearest neighbour
    (_, idx) = kdtree.query(gamma[chosenDoc,:], 1)
    
    print 'Tweet 1: ' + origDocs[chosenDoc]
    print 'Tweet 2: ' + origDocs[idx]    
    text = 'Do these two tweets discuss the same specific need, problem or emergency, or are they both irrelevant to rescue/aid workers? Enter Y or N'
    
    label=0
    while label==0:        
        resp = raw_input(text)
        if resp.lower()=='y':
            label = 1
        elif resp.lower()=='n':
            label = -1
        else:
            print 'Need to enter Y or N'    
    
    #now modify gamma
    g1 = gamma[chosenDoc,:]
    g2 = gamma[idx,:]
    
    print 'Topic weights for first tweet (before): '
    g1z = zip(g1, range(0, len(g1)))
    topics = sorted(g1z, key = lambda x: x[0], reverse=True)
    if len(g1z) > 5:
        nTopicsToShow = 5
    else:
        nTopicsToShow = len(g1z)
    for i in range(0, nTopicsToShow):
        print str(topics[i][1]) + ', ' + str(topics[i][0])
    print 'Topic weights for second tweet (before): '
    g2z = zip(g2, range(0, len(g1)))
    topics = sorted(g2z, key = lambda x: x[0], reverse=True)
    for i in range(0, nTopicsToShow):
        print str(topics[i][1]) + ', ' + str(topics[i][0])
                
    gdiff = g1 - g2
    gdiff = gdiff / 3
    g1 = g1-(label*gdiff)
    g2 = g2+(label*gdiff)

#     scale1 = numpy.sum(g1)
#     scale2 = numpy.sum(g2)
#     theta1 = numpy.divide(g1, scale1)
#     theta2 = numpy.divide(g2, scale2)

#this one sort of worked-ish    
#     if label==1:
#         g1 = numpy.multiply(theta1,numpy.power(theta2,0.2))
#         g1 = g1 * scale1 / numpy.sum(g1)
#         g2 = numpy.multiply(numpy.power(theta1,0.2),theta2)
#         g2 = g2 * scale2 / numpy.sum(g2)
#     else:
#         g1 = numpy.multiply(theta1,numpy.power(1-theta2,0.2))
#         g1 = g1 * scale1 / numpy.sum(g1)
#         g2 = numpy.multiply(numpy.power(1-theta1,0.2),theta2)
#         g2 = g2 * scale2 / numpy.sum(g2)
    
#     merged = numpy.sqrt( numpy.multiply(g1,g2) )
#     labellerError = 0.2
#     merged = merged * (1-labellerError)
#     g1 = numpy.divide(g1 + label*merged, 2-labellerError)
#     g2 = numpy.divide(g2 + label*merged, 2-labellerError)
    
#     g1 = numpy.multiply(g1, scale1)
#     g2 = numpy.multiply(g2, scale2)
    
    gamma[chosenDoc,:] = g1
    gamma[idx,:] = g2
    
    print 'Topic weights for first tweet (after): '
    g1z = zip(g1, range(0, len(g1)))
    topics = sorted(g1z, key = lambda x: x[0], reverse=True)
    for i in range(0, nTopicsToShow):
        print str(topics[i][1]) + ', ' + str(topics[i][0])
    print 'Topic weights for second tweet (after): '
    g2z = zip(g2, range(0, len(g1)))    
    topics = sorted(g2z, key = lambda x: x[0], reverse=True)
    for i in range(0, nTopicsToShow):
        print str(topics[i][1]) + ', ' + str(topics[i][0])
    
    return gamma

def interactCategories(nInts, post, catNames, origDocs, catIdxs, gamma=None):
    
    #Select objects to label -> choose those furthest from existing, i.e. most uncertain
    #if gamma==None:
    #    H = -numpy.sum(numpy.multiply(post, numpy.log(post)), 1)
    #else:
    #    H = topicEntropy(gamma)
    H = -numpy.sum(numpy.multiply(post, numpy.log(post)), 1)
    
    labels  = []
    idxs = []
    
    for i in range(nInts):
    
        idx = numpy.argmax(H)
        H[idx] = float('-inf') 
        
        #Ask the user for a categorical label
        #print 'Tweet: ' + origDocs[idx]
        #print 'Tweet to label: ' + str(idx)
        text = 'Choose a category for the object (enter a number): '
        catIdx = 0
        for cat in catNames:
            text = text + '(' + str(catIdx) + ') ' + cat
            catIdx += 1
        
        label = catIdxs[idx]
        
#Uncomment to get real labels         
#         label=-1
#         while label==-1:        
#             resp = raw_input(text)
#             try: 
#                 label = int(resp)
#             except ValueError:
#                 print 'Need to pick a number'

        labels.append(label[0,0])
        idxs.append(idx)
    
    return (labels, idxs)      
      
def interactCatLdaPush(nInts, post, catNames, origDocs, docset, catIdxs, lda, gamma):      
    (newTrLabels, newTrIdxs) = interactCategories(nInts, post, catNames, origDocs, catIdxs, gamma)
    for l in range(nInts):
        for m in range(nInts):
            if l==m:
                continue
            direction = 0
            if newTrLabels[l]==newTrLabels[m]:
                direction = 1
            else:
                direction = -1
            lda.insert_push(newTrIdxs[l], newTrIdxs[m], direction)
            
    (_, gamma, lda) = updateLda(lda, docset, gamma)
    return (gamma, newTrLabels, newTrIdxs)

def interactCatLdaTop(nInts, post, catNames, origDocs, docset, catIdxs, lda, gamma):      
    (newTrLabels, newTrIdxs) = interactCategories(nInts, post, catNames, origDocs, catIdxs, gamma)
    
    for d, l in zip(newTrIdxs, newTrLabels):      
        total = sum(gamma[d,:])       
        lda.insert_known_topic(d, l, total, gamma.shape[0])        
            
    (_, gamma, lda) = updateLda(lda, docset, None) # don't pass in gamma as we need to restart the process
    return (gamma, newTrLabels, newTrIdxs)
              
def printAucs(nClasses, post, testLabels, classifierName=' '):
    
    aucs = []
    print ' --- AUCS for classifier ' + classifierName + ' --- '
    for j in range(nClasses):
        predictions = post[:,j]
        testLabels_j = testLabels==j
        if sum(testLabels_j) == 0:
            print 'No test labels of class ' + str(j)
            continue
        #else:
        #    print 'Class ' + str(j) + ' has ' + str(sum(testLabels_j)) + 'labels'
        (fpr, tpr, _) = roc_curve( testLabels_j.tolist(), predictions.tolist())
        auc_nb_j = auc(fpr, tpr)
        aucs.append(auc_nb_j)
        print 'Class' + str(j) + ': ' + str(auc_nb_j)
    
    return aucs
    
def knnClassify(trData, trLabels, testData, nClasses):
    #run the knn classifier using the raw features
    (nDocs, _) = testData.shape
    knnLabels = numpy.zeros((nDocs, nClasses))
    
    leafsize = 5
    kdtree = KDTree(trData.tolist(), leafsize)
    (_, idxs) = kdtree.query(testData.tolist(), 3)
    for d in range(nDocs):
        testPointIdxs = idxs[d]
        votes = [trLabels[l] for l in testPointIdxs]
        classVotes = [0] * nClasses
        for v in votes:
            classVotes[v] += 1
        knnLabels[d, classVotes.index(max(classVotes))] = 1
    return (knnLabels, kdtree)
    
def nbClassify(trData, trLabels, testData, nClasses):
    
    logL = calcWordLikelihoods(trData, trLabels, nClasses) #obtain a vector where rows are classes, columns are words
    logP = calcClassPriors(trLabels, nClasses) # obtain a vector where columns are classes  
    post = calcDocPosteriors(logL, logP, testData) #obtain a vector where rows are docs, columns are classes
    
    return post
    
def calcWordLikelihoods(trData, trLabels, nClasses):
    
    (nDocs, nFeats) = trData.shape
    
    counts = numpy.ones((nClasses,nFeats))
    
    for d in range(nDocs):
        l = trLabels[d]
        counts[l,:] = counts[l,:] + trData[d,:]
       
    totals = numpy.sum(counts, 0)
    for l in range(nClasses): 
        counts[l,:] = numpy.divide(counts[l,:], totals)
        
    return numpy.log(counts) 

def calcClassPriors(trLabels, nClasses):
    
    P = numpy.ones((1, nClasses))
    
    for j in range(nClasses):
        P[:,j] += numpy.sum(trLabels==j)
        
    P = numpy.divide(P, numpy.sum(P,1))
    
    return numpy.log(P)

def calcDocPosteriors(logL, logP, testData):
    
    (nDocs, _) = testData.shape
    
    (_, nClasses) = logP.shape
    
    logLdoc = numpy.zeros((nDocs,nClasses))
    
    for d in range(nDocs):
        doc = testData[d,:]
                
        for j in range(nClasses):
            logLdoc[d,j] = numpy.sum(numpy.multiply(logL[j,:], doc))
        
        logLdoc[d,:] = logLdoc[d,:] + logP
    
    postDoc = numpy.zeros((nDocs,nClasses))    
    Ldoc = numpy.exp(logLdoc)
    for j in range(nClasses):
        postDoc[:, j] = numpy.divide(Ldoc[:, j], numpy.sum(Ldoc, 1))
    
    return postDoc
        
def getRawFeatures(docset, vocab):
    
    nWords = len(vocab)
    
    #feature matrix with first index corresponding to the document, second to the word
    F = numpy.zeros((len(docset), nWords), numpy.int8)
    for i in range(len(docset)):
        d = docset[i]
        for w in d:
            wIdx = vocab.index(w)
            F[i, wIdx] += 1
                
    return F
    
def runUnsupervisedLda(docset, vocab, nIts=100, renewLda=False, K=100):
    """
    Load and analyses a bunch of tweets using online VB for LDA.
    """
    
    filename_lambda = '/home/edwin/git/TWeetAnalysisTest/data/lambda-final.dat'
    filename_gamma = '/home/edwin/git/TWeetAnalysisTest/data/gamma-final.dat'
    
    # The total number of documents to analyse
    D = len(docset)

    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K+1, 1./K+1, 1024., 0.)
    
    if renewLda==False:
        try:
            gamma = numpy.loadtxt(filename_gamma)
#             olda = numpy.loadtxt(filename_lambda)
            import cPickle
            with open(filename_lambda, 'rb') as input:
                olda = cPickle.load(input)
            lambda_lda = olda._lambda
            return (lambda_lda, gamma, olda)
        except IOError, e:
            print 'No LDA files available, so re-running LDA: ' + str(e)
    
    (olda._lambda, gamma, olda) = updateLda(olda, nIts, docset)
    
    # Save lambda, the parameters to the variational distribution over topics, 
    # and gamma, the variational hyperparameters over topic weights for the articles 
    import cPickle
    with open(filename_lambda, 'wb') as output:
        cPickle.dump(olda, output, -1)
#     numpy.savetxt(filename_lambda, olda)
    numpy.savetxt(filename_gamma, gamma)    
    
    return (olda._lambda, gamma, olda)
    
def updateLda(olda, docset, gamma=None, nIts=10):
    
    D = olda._D
    
    for iteration in range(0, nIts):

        # Run online LDA with current document set
        (gamma, bound) = olda.update_lambda(docset, gamma)
        # Compute an estimate of held-out perplexity
        (_, wordcts) = onlineldavb.parse_doc_list(docset, olda._vocab)
        perwordbound = bound * len(docset) / (D * sum(map(sum, wordcts)))
        print '%d:  held-out perplexity estimate = %f' % \
            (iteration, numpy.exp(-perwordbound))

    return (olda._lambda, gamma, olda)

def getVocab(docset):
    # Obtain a set of stopwords that we know are uninformative a priori
    stopfile = '/homes/49/edwin/local/git/TWeetAnalysisTest/data/stopwords.txt'
    stopfileh = open(stopfile, 'r')
    stopwords = stopfileh.read() 
    stopwords = stopwords.split(',')   
    
    # File to write the vocabulary (the complete set of observed features/words)
    vocabfile = '/homes/49/edwin/local/git/TWeetAnalysisTest/vocab.txt' 
    vocab = []
    vocabfileh = open(vocabfile,'w')
    
    preprocDocset = []
    
    for d in docset:
        
        preprocDoc = []
        
        for w in d.split():
            w = ''.join([c for c in w if c not in ('!','?','(',')',',','.','"',':',';',"'",'…','…','…')])
            w = w.lower()
            w = re.sub(r'[^a-z]', '', w)
            if (not w in stopwords) and (len(w)>2):
                if w not in vocab:
                    vocab.append(w)
                preprocDoc.append(w)
        preprocDocset.append(preprocDoc)
    for w in vocab:
        vocabfileh.write(w)
        vocabfileh.write('\n')
    vocabfileh.close()
    
    return (vocab, preprocDocset)

def getSampleTweets():
    docset = []
    articlenos = []
    cats = []
    
    tweetfilename = '/home/edwin/git/TWeetAnalysisTest/data/tweets_cats.csv'
    tweetfile = open(tweetfilename,'r')
    lines = list(tweetfile)
    for l in lines[1:]:
        toks = l.split(',',1)
        if len(toks)>1 and '"' in toks[1]:
            try: 
                tweetid = float(toks[0])
            except ValueError:
                tweetid = -1
                
            if tweetid >= 0:
                
                catToks = toks[1].split('"')
                
                if len(catToks) < 4:
                    cats.append('?')
                else:
                    catToks3 = catToks[3].split('-')
                    cats.append(catToks3[0])
                    
                docset.append(catToks[1])
                
                articlenos.append(tweetid)
                #print 'new tweet:' + str(len(docset))
            else:
                print 'Something went wrong with tweet ID reading: ' + str(l)
        elif len(toks)>0:
            print 'Something went wrong with number of tokens: ' + str(l)  
        
    catNames = list(set(cats))
    catIdxs = []
    for catVal in cats:
        catIdxs.append(catNames.index(catVal))
        
    (vocab, preprocDocset) = getVocab(docset)
        
    return (vocab, preprocDocset, docset, articlenos, catIdxs, catNames)

if __name__ == '__main__':
    main()
