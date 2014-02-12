#!/usr/bin/python

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

import sys, random, math, numpy as n, csv

def main():
    print 'Receiving locations of targets and simulating sets of crowd reports.'
    if len(sys.argv)<5:
        print 'Need to supply filenames as arguments for files containing tweet templates and events!'
        print 'For now we will generate 10 random targets.'
        nTargets = 10
        nTypes = 5
        targets = genRandomTargets(nTypes-1, nTargets, 100, 100) # nTypes-1 because we leave one index free for irrelevant tweets
    else:
        targetFile = sys.argv[4]
        with open(targetFile, 'rb') as targetInput:
            tReader = csv.reader(targetInput, delimiter=',')
            targets = n.matrix([])
            for row in tReader:
                targets = n.concatenate([targets,row])
#             targets = cPickle.load(input)
        nTargets = targets.shape[0]
        types = set(targets[:,0])
        nTypes = len(types)
                        
    templateFile = sys.argv[1]    
    (reports, reportLocs, reportTags) = generateReports(targets, templateFile, nTypes)
    tweetFile = sys.argv[3]
    outputFile = sys.argv[2] #load any older reports from the tweet file. 
    #If target is no longer listed, related reports can be deleted from the map. 
    #Otherwise we re-run the clustering to find the event and keep the tweets.
    
    (oldTweets, oldTweetLocs, oldTweetTags) = loadOldTweets(tweetFile)
        
    oldCorrespReps = loadOldEvents(outputFile)
    
    if oldCorrespReps != None and oldTweets != None:
        (oldTweets, oldTweetLocs, oldTweetTags) = removeCompletedTargets(oldCorrespReps, \
                                                                     oldTweets, oldTweetLocs,\
                                                                     oldTweetTags)
        if len(oldTweets) > 0:
            reports = n.concatenate([oldTweets, reports])
            reportLocs = n.concatenate([oldTweetLocs, reportLocs])
            reportTags = n.concatenate([oldTweetTags, reportTags])
        
    (eventTypes, eventLocs, eventStrength, correspReports) = \
                                                predictEvents(reportLocs, reportTags, nTypes)
        
    writeTweetFile(tweetFile, reports, reportLocs, reportTags)
    writeEventFile(outputFile, eventTypes, eventStrength, eventLocs, correspReports)
    print 'Written tweets and events to the data files.'
        
def writeTweetFile(tweetFile, reports, reportLocs, reportTags):
    with open(tweetFile, 'wb') as output:
        repWriter = csv.writer(output, delimiter=',')
        for idx, r in enumerate(reports):
            row = [r, reportLocs[idx,0], reportLocs[idx,1]]
            for i in range(reportTags.shape[1]):
                row.append(reportTags[idx,i])
            repWriter.writerow(row)
            
def writeEventFile(outputFile, eventTypes, eventStrength, eventLocs, correspReports):            
    with open(outputFile, 'wb') as output:     
        repWriter = csv.writer(output, delimiter=',')
        for idx, e in enumerate(eventTypes):
            eventRow = [e, eventStrength[idx], eventLocs[idx,0],  eventLocs[idx,1]]
            reps = list(correspReports[idx])
            [eventRow.append(reps[i]) for i in range(len(reps))]
            repWriter.writerow(eventRow)            
        
def loadOldEvents(outputFile):        
    try:
        with open(outputFile, 'rb') as eventInput:
            oldCorrespReps = []
            evReader = csv.reader(eventInput, delimiter=',')
            for row in evReader:
                [oldCorrespReps.append(int(row[i])) for i in range(4,len(row))]
    except IOError:
        print 'No old reports to load'
        oldCorrespReps = None   
        
    return oldCorrespReps        
        
def loadOldTweets(tweetFile):
    
    oldTweets = []
    oldTweetLocs = None
    oldTweetTags = None
    
    try:
        with open(tweetFile, 'rb') as tweetInput:
#             (oldTweets, oldTweetLocs, oldTweetTags, oldEvents) = cPickle.load(input)
            tweetReader = csv.reader(tweetInput, delimiter=',')
            
            nRows = 0
            for row in tweetReader:
                
                oldTweets.append(row[0])
                if oldTweetLocs==None:
                    oldTweetLocs = n.matrix([float(row[1]),float(row[2])])
                else:
                    oldTweetLocs = n.concatenate([oldTweetLocs,n.matrix([float(row[1]),float(row[2])]) ])
                if oldTweetTags==None:
                    tags = []
                    [tags.append(float(row[r])) for r in range(3,len(row)) ]
                    oldTweetTags = n.matrix(tags)
                else:
                    tags = []
                    [tags.append(float(row[r])) for r in range(3,len(row)) ]
                    oldTweetTags = n.concatenate([oldTweetTags,n.matrix(tags) ])
            nRows +=1
    except IOError:
        print 'No old reports to load'
        oldTweets = None
    except SyntaxError:
        print 'Syntax Error: ' + str(nRows) 
        oldTweets = None  
        
    return (oldTweets, oldTweetLocs, oldTweetTags)      
        
def removeCompletedTargets(correspReps, oldTweets, oldTweetLocs, oldTweetTags):
    #OLD EVENTS should only contain the remaining events. Anything completed
    #should be removed from that list. So remove any corresponding tweets that now have
    # no topic (we ignore any irrelevant tweets from previous iterations). 
    #Get IDs of remaining tweets.
    #Return set of old tweets minus those relating to expired targets.
    print 'Removing old tweets relating to events that are not listed in the events output file.'
    
    nTweets = len(oldTweets)
    tweetIdxs = range(nTweets)
    
    oldTweetsFiltered = []

    stillRelevant = []
    for i in tweetIdxs:
        if i in correspReps:
            oldTweetsFiltered.append(oldTweets[i])
            stillRelevant.append(i)
            
    oldTweetLocs = oldTweetLocs[stillRelevant, :]
    oldTweetTags = oldTweetTags[stillRelevant, :]
    
    return (oldTweetsFiltered, oldTweetLocs, oldTweetTags)    
    
    
def genRandomTargets(nTypes, nTargets, maxX, maxY):
    targets = n.zeros((nTargets, 3))
    
    for i in range(nTargets):
        targets[i,0] = random.randrange(nTypes)
        targets[i,1] = random.random()*maxX
        targets[i,2] = random.random()*maxY
        
    print 'Generated random targets with types: ' + str(targets[:,0])
        
    return targets
    
def generateReports(targets, templateFile, nTypes):
    print 'generating reports...'
    
    templates = {}
    
    with open(templateFile, 'rb') as csvfile:
        templateReader = csv.reader(csvfile, delimiter=',')
        for row in templateReader:
            if int(row[0]) not in templates:
                templates[int(row[0])] = []
            templates[int(row[0])].append(row[1])    
    
    #for each event in the list, generate an appropriate set of tweets
    
    reports = []
    repLocs = None
    repTags = None #in real test apps this will be filled in by IBCC
    
    for idx, t in enumerate(targets[:,0]):
        
        t = int(t)
        
        targetTemplates = templates[t]
        
        nTweets = random.randint(5,20)
        for _ in range(nTweets):
            templTweet = random.randrange(len(targetTemplates))
            reports.append(targetTemplates[templTweet])
            
            tagDist = [random.random() for _ in range(nTypes)]
            sumDist = n.sum(tagDist)
            tagDist[t] += sumDist
            tagDist /= (sumDist*2)
            tagDist = n.matrix(tagDist)
            if repTags == None:
                repTags = tagDist
            else:
                repTags = n.concatenate([repTags, tagDist])
            
            locX = targets[idx,1] + random.uniform(-10,10)
            locY = targets[idx,2] + random.uniform(-10,10)  
            if repLocs==None:
                repLocs = n.matrix([locX, locY])          
            else:
                repLocs = n.concatenate([repLocs, n.matrix([locX, locY])])
        
    #throw in irrelevant tweets. These won't be displayed directly on map, but can be shown 
    # in a side bar to show how we filter the data. They are discarded at the next round.
    targetTemplates = templates[10] #index 10 for irrelevant rubbish
    nTweets = random.randint(20,50)
    t = nTypes-1 #The last type is the irrelevant shite
    for _ in range(nTweets):
        templTweet = random.randrange(len(targetTemplates))
        reports.append(targetTemplates[templTweet])
        
        tagDist = [random.random() for _ in range(nTypes)]
        sumDist = n.sum(tagDist)
        tagDist[t] += sumDist
        tagDist /= (sumDist*2)
        tagDist = n.matrix(tagDist)
        repTags = n.concatenate([repTags, tagDist])
        
        locX = targets[idx,1] + random.uniform(-10,10)
        locY = targets[idx,2] + random.uniform(-10,10)            
        repLocs = n.concatenate([repLocs, n.matrix([locX, locY]) ])  
    
    return (reports, repLocs, repTags)
        
def predictEvents(reportLocs, reportTags, nTypes):
    #Given the tweets, we should be able to find noisy estimates of the epicentre 
    # of the events, e.g. for each event type, we can create an intensity map across the
    #whole area. The peaks (thresholded) can be marked on the map as possible events.
    #Alternatively, we can cluster the tweets of each type into events. Cluster size 
    #then corresponds to probability and seriousness of event.
    # We can eventually build the cluster model into IBCC as an additional layer of hierarchy:
    # target classes are now events; location must be accounted for; unreliability of tweet
    # as well as labellers must be accounted for. 
    # The event prediction/clustering should relate more to distance of 
    # tweets than the no. clusters required.
    # The event must also account for the p(tag) for each of the tweets, i.e. their relevance
    # may vary and their location may be shaky. 
    #We can zoom in to split the event probabilities (show multiple modes) -- don't bother
    # with this here. This zooming is necessarily
    #part of clustering because clustering can either be hierarchical or we must select
    # some number of clusters or level at which we consider items to be grouped. 
    
    #Output a set of events with possible locations and links to related tweet ids 
    #(ordered by relevance to the event). 
    print 'Clustering tweets to find events... TODO: Ignore the tags marking tweets as irrelevant'
    
    eventLocs = None 
    eventStrength = []
    eventTypes = []
    correspReports = []
    

    #Either: find initial cluster centres by adding a new cluster when a point is too far away.
    #cheat by using kmeans with no. targets as the number of clusters. Can use
    #kmeans to do iterative bit once no. clusters and their centres are estimated using above idea.
    
    nReports = reportLocs.shape[0]
    
    for t in range(nTypes-1):
        theta = 10 #distance to consider merging
        theta2 = 0.1 # distance to assume equivalence -> merge completely
        theta3 = 0.1 # tag probability value to assume tweet is not relevant to topic    
    
        #Initialise cluster centres
        locs = n.copy(reportLocs)
        strength = n.copy(reportTags[:,t])
        members = []
        for e in range(nReports):
            members.append(set([e]))
        
        nMaxIts = 30;
        
        for _ in range(nMaxIts):
            theta += 1
            nMerges = 0;
        
            nClusters = 0;
        
            #merge adjacent clusters
            for e in range(nReports):
            
                if strength[e]<theta3: #already been merged with another event or is not relevant to target
                    strength[e] = 0
                    continue;
                else:
                    nClusters += 1;
            
                dists = n.zeros((nReports,1))
            
                for f in range(nReports):
                
                    if e==f or strength[f]<theta3: #already been merged
                        continue
                
                    dists[f] = euclidDist(locs[e,:], locs[f,:])
            
                distToNeighbours = 0
                neighbourToMerge = []
                for f in range(nReports):
                    if e!=f and strength[f]>=theta3 and dists[f]<theta: #find total to merge with
                        if dists[f]<theta2:
                            dists[f] = 1
                            distToNeighbours = 1
                            neighbourToMerge = [f]
                            break
                        else:
                            distToNeighbours += 1/dists[f]
                            neighbourToMerge.append(f)

                #for f in range(nReports):
                for f in neighbourToMerge:
                    totalEStrength = strength[e]
            
                    #if e!=f and dists[f] < theta and strength[f]>0:
                
                    #portion to merge with f
                    portion = 1/(dists[f]*distToNeighbours)
                    eStrength = totalEStrength * portion
            
                    locs[f,0] = (locs[e,0]*eStrength + locs[f,0]*strength[f])/ (eStrength+strength[f])
                    locs[f,1] = (locs[e,1]*eStrength + locs[f,1]*strength[f])/ (eStrength+strength[f])
                    strength[f] = eStrength + strength[f]
                    strength[e] = strength[e] - eStrength
                    [members[f].add(m) for m in members[e] ]
                    
#                     print 'Membership of ' + str(f) + ' is now ' + str(len(members[f]))
                    nMerges+=1
#             print 'No. merges this iteration: ' + str(nMerges) + ' with no. clusters=' + str(nClusters)
#             print 'Weakest cluster: ' + str(min(strength))
            if nMerges==0:
                break
        
        liveEvents = []
        for idx, s in enumerate(strength):
            if s>0.5: # more likely than not to be an event if one tweet has suggested the event
                liveEvents.append(idx)
            
        nEventsFound = len(liveEvents)
        print 'We found ' + str(nEventsFound) + ' events by clustering tweets of type ' + str(t)
        if eventLocs==None:
            eventLocs = locs[liveEvents,:]
        else:
            eventLocs = n.append(eventLocs,locs[liveEvents,:], 0)
            
        [eventTypes.append(t) for _ in range(nEventsFound)]
        eventStrength = n.append(eventStrength, strength[liveEvents])
        [correspReports.append(members[l]) for l in liveEvents]
    
    return (eventTypes, eventLocs, eventStrength, correspReports)
#     for r in range(reportLocs.shape[0]):    
#         nEvents = len(events.shape[0])
#         
#         loc = reportLocs[r,:] # pair of x y coords
#         tags = reportTags[r,:] #distribution over tags
#         
#         idx of corresponding event
#         if nEvents>0:
#             eDists = n.zeros((nEvents, 1))
#         
#         for e in range(nEvents):
#             event = events[e]
#             d = euclidDist(loc, event[1:2]) #need to implement this
#             eDists[e] = d
#             
#         eDists = n.divide(eDists, tags[events[:,0]])
#             
#             update existing clusters: add counts to them proportional to the tag weight
#              and inverse distance
#                 
#                 update the cluster locations according to the inverse distance * tag weight
#         if min(eDists) > theta:
#                 add new clusters for each tag type with p(tag|report)>0
#                 
#             newEvent = [, loc]
#             events = n.concatenate(events, newEvent)
#         else:
            #add new clusters for the tag types
            
        #At the end there may be some events with very low counts. Cut them right out. 
        
        # Normalise the others somehow. 
        #To avoid this, we could cheat and find the real number of each type of target,
        #then apply kmeans (hope it doesn't look too accurate). The weights of each target
        #can be found most simply by no. tweets. Distribution may also tell us something 
        #e.g highly concentrated versus very spread out) but ignore for now.   
      
def euclidDist(loc1, loc2):
    sqDist = (loc1-loc2)*n.matrix((loc1-loc2)).T
    return math.sqrt(sqDist[0])
      

if __name__ == '__main__':
    main()
