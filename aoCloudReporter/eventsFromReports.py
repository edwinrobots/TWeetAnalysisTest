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

import sys, random, math, numpy as n, csv, re
from sklearn.cluster import DBSCAN

def main():
    print 'Receiving tweets with locations and clustering to find events.'
    print 'Will detect and write a set of events at each iteration, i.e. will re-run clustering as each report appears'

    tweetFile = sys.argv[1]
    
    reports = []
    reportLocs = []
    reportTags = []
    
    with open(tweetFile, 'rb') as tweetInput:
        tReader = csv.reader(tweetInput, delimiter=',')
        
        for row in tReader:
            try:
                float(row[0])
            except ValueError:
                continue;
            
            reports.append(int(row[0]))
            if len(reportLocs)==0:
                reportLocs = n.matrix([float(row[4]), float(row[5]) ])
            else:
                reportLocs = n.concatenate( (reportLocs, n.matrix([float(row[4]), float(row[5])]) ) )
            reportTags.append(row[1])
         
    types = getTypeList(reportTags)                    
                        
    outputFile = sys.argv[2] 
        
    events = []
        
    for it in range(len(reports)):
        
        currReps = reports[0:it+1]
        currRepLocs = reportLocs[0:it+1]
        currRepTags = reportTags[0:it+1]
        
        #Only update every 10 iterations
        if it % 5 != 0:
            continue             
        
        (currRepTags, currRepIdxsByType) = convertClassifications(currReps, currRepTags, types)        
        
        currentEvents = predictEvents(it, currReps, currRepLocs, currRepTags, types, currRepIdxsByType)
        if currentEvents:
            events += currentEvents
        
    writeEventFile(outputFile, events)
    print 'Written tweets and events to the data files.'
       
def convertClassifications(reportIds, reportTags, types):
    
    reportIdxsByType = {}
    
    for t in range(len(types)):
        reportIdxsByType[types.keys()[t]] = []
    
    for i, s in enumerate(reportTags):
        starts = re.findall('[0-9][a-z]?', s)
        
        reportTags[i] = []
        for code in starts:
            if len(code)>1:
                #we have a more detailed code -- add the higher level too
                reportTags[i].append(code[0])
                reportIdxsByType[code[0]].append(i)

            #get the complete code's id
            if not code in types.keys():
                continue
            
            reportTags[i].append(code)
            reportIdxsByType[code].append(i)

    return reportTags, reportIdxsByType
        
def getTypeList(tags):
    print "Using hard-coded types for Ushahidi Haiti project"
    types = {}
    types["1"] = "Emergency"
    types["1a"] = "Emergency -- Trapped People"
    types["1b"] = "Emergency -- Medical Emergency"
    types["1d"] = "Emergency -- Fire"
    types["2"] = "Vital Lines"
    types["2a"] = "Vital Lines -- Contaminated Water"
    types["2b"] = "Vital Lines -- Water Shortage"
    types["2c"] = "Vital Lines -- Power Outage"
    types["2d"] = "Vital Lines -- Shelter Needed"
    types["2e"] = "Vital Lines -- Food Shortage"
    types["2f"] = "Vital Lines -- Security Concern"
    types["2g"] = "Vital Lines -- Fuel Shortage"
    types["3"] = "Public Health"
    types["3a"] = "Public Health -- Medical and Health Supplies Needed"
    types["3b"] = "Public Health -- Infectious Human Disease"
    types["3d"] = "Public Health -- OBGYN/Women's Health"
    types["4"] = "Menaces and Security Threats"
    types["4a"] = "Menaces and Security Threats -- Looting"
    types["5"] = "Infrastructure Damage"
    types["5a"] = "Infrastructure Damage -- Collapsed Building"
    types["5b"] = "Infrastructure Damage -- Roads Blocked"
    types["6"] = "Natural Hazards"
    types["6a"] = "Natural Hazards -- After Shock"
    #types["6c"] = "Natural Hazards -- Missing People"
    #types["6d"] = "Natural Hazards -- Asking to Forward a Message"
    types["7"] = "Services Available"
    types["7a"] = "Services Available -- Clinics and Hospitals Operating"
    types["7b"] = "Services Available -- Food Distribution Point"
    types["7c"] = "Services Available -- Non-Food Aid Distribution Point"
    types["7d"] = "Services Available -- Rubble Removal"
    types["7e"] = "Services Available -- Human Remains Management"
    types["8"] = "Other"
    types["8a"] = "Other -- Search and Rescue"
    types["8b"] = "Other -- Person News"
    types["8c"] = "Other -- IDP Concentration"
    
    return types 
         
def writeEventFile(outputFile, events):            
    import json
    with open(outputFile, 'w') as output:
        json.dump(events, output, indent=2)
   
def predictEventsOfType(reportIds, reportLocs, typeId, it, types):
    print "Iteration " + str(it) + " | Predicting events of type " + str(typeId)
    
    #Initialise cluster centres
    locs = n.copy(reportLocs)
    #print("assuming a strength of 1 for all nodes")
#     strength = n.ones((len(reportIds), 1))

    newEvents = {}
    db = DBSCAN(eps=0.04, min_samples=3).fit(reportLocs)
    coreSamples = db.core_sample_indices_
    labels = db.labels_
     
    for idx in coreSamples:
        
        label = labels[idx]
        
        if not label in newEvents:
            newEvents[label] = {}
            newEvents[label]["REPORTIDS"] = []
            newEvents[label]["LAT"] = 0
            newEvents[label]["LON"] = 0
        
        newEvents[label]["LAT"] += reportLocs[idx,0]
        newEvents[label]["LON"] += reportLocs[idx,1]
        newEvents[label]["REPORTIDS"].append(reportIds[idx])
            
    for l in newEvents.keys():
        
        nMembers = len(newEvents[l]["REPORTIDS"])
        
        newEvents[l]["LAT"] /= nMembers
        newEvents[l]["LON"] /= nMembers

        newEvents[l]["STRENGTH"] = nMembers
        
        newEvents[l]["CATEGORY"] = typeId + ". " + types[typeId] 
        newEvents[l]["TIMES"] = it
        
    return newEvents.values()

def predictEvents(it, reportIds, reportLocs, reportTags, types, reportIdxsByType):
    currentEvents = []
    for t in types.keys():
        idxs_t = reportIdxsByType[t]
        if len(idxs_t)>=5:            
            reportLocs_t = n.concatenate([reportLocs[i] for i in idxs_t])
            currentEvents += predictEventsOfType([reportIds[i] for i in idxs_t], \
                                                 reportLocs_t, t, it, types)

    return currentEvents 

if __name__ == '__main__':
    main()
