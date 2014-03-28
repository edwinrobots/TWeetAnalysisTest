function initialize() {

  mapCentre = new google.maps.LatLng(19, -72.5) //52.953250, -1.187569)

  var mapOptions = {
    zoom: 9,
    center: mapCentre,
    mapTypeId: google.maps.MapTypeId.SATELLITE
  };

  var map = new google.maps.Map(document.getElementById('map-canvas'),
      mapOptions);

  markers = {}
  openWindows = {}
  
//   for (var id in reportTitles){  
//     timedReport(id, map, markers, openWindows)
//   }
//   
  eventMarkings = {}
   
  for (var idx in eventTimes){  
    timedEvent(idx, map, eventMarkings, openWindows, eventTimes[idx])
  }
}

/*Need a function that reads an update from file, then either adds or removes the corresponding marking. 
This requires us to keep a list of markings.*/

function timedReport(id, map, markers, openWindows){
    setTimeout(function(){
        drawRep(id, map, new google.maps.LatLng(reportLat[id], reportLon[id]), markers, openWindows)
    }, reportTimes[id]*1000)
    
    setTimeout(function(){
        removeEvent(markers[id])
    }, reportTimes[id]*1000 + 20000)
}

function timedEvent(id, map, markers, openWindows, startTime){
    setTimeout(function(){
        drawEvent(id, map, new google.maps.LatLng(eventLat[id], eventLon[id]), eventStrength[id], eventCat[id], markers, openWindows)
    }, startTime*1000)
    
    setTimeout(function(){
        removeEvent(markers[id])
    }, (startTime+5)*1000)
}

function removeEvent(marking){
    var l = marking.length

    for (var i=0; i<marking.length; i++){
        marking[i].setMap(null);
    }
}

function addReportInfo(infoBox, reportId){
    
    type = reportCats[reportId]
    
    cats = "<br/><b>Categories: </b>"
    for (var t=0; t<type.length; t++){
        cats += "<br/>" + type[t]
    }
    
    certainty = Math.random()/2 + 0.5;
    certainty = certainty.toFixed(1);
    
    infoBox.content = infoBox.content + "<b>Title: </b>" + reportTitles[reportId] + cats + "<br/><b>Approx. certainty: </b>" + certainty;
    
}

function addEventInfo(infoBox, id){
    relReps = eventReports[id]
    reportList = ''
    for (var r in relReps){
        reportList = reportList + "<br/><br/>" + r + ". " + reportTitles[relReps[r]];
        if (r >= 10) break
    }
    infoBox.content += reportList
}

function drawRep(id, map, loc, markers, openWindows){
        
    markers[id] = new google.maps.Marker({
            position: loc,
            map: map,
            title: 'Event!'
        });
    google.maps.event.addListener(markers[id], 'click', function() {
                    
        var infoWindow = new google.maps.InfoWindow({
            content: ''
        });
        addReportInfo(infoWindow,id)         
        
            infoWindow.open(map,markers[id]);
            for (var i in openWindows){
                
                openWindows[i].close();
                delete openWindows[i];
            }
            openWindows[id] = infoWindow;
        });
    return markers[id]
}

function getEventColour(cat){
    if (cat=="1."){
        col = '#FF0000'
        img = "http://maps.google.com/mapfiles/kml/shapes/caution.png"
    }
    else if (cat=="1a."){
        col = '#FF0000'
        img = "http://maps.google.com/mapfiles/kml/shapes/falling_rocks.png"
        
    }
    else if (cat=="1b."){
        col = '#FF0000'
        img = "http://maps.google.com/mapfiles/kml/shapes/hospitals.png"
    }
    else if (cat=="1d."){
        col = '#FF0000'
        img = "http://maps.google.com/mapfiles/kml/shapes/firedept.png"
    }
    else if (cat=="2."){
        col = '#00FF00'
        img = "http://maps.google.com/mapfiles/kml/shapes/water.png"
    }
    else if (cat=="2a."){
        col = '#00FF00'
        img = "http://maps.google.com/mapfiles/kml/shapes/water.png"
    }
    else if (cat=="2b."){
        col = '#00FF00'
        img = "http://maps.google.com/mapfiles/kml/shapes/water.png"
    }
    else if (cat=="2c."){
        col = '#00FF00'
        img = "http://maps.google.com/mapfiles/kml/shapes/thunderstorm.png"
    }
    else if (cat=="2d."){
        col = '#00FF00'
        img = "http://maps.google.com/mapfiles/kml/shapes/campground.png"
    }
    else if (cat=="2e."){
        col = '#00FF00'
        img = "http://maps.google.com/mapfiles/kml/shapes/convenience.png"
    }
    else if (cat=="2f."){
        col = '#00FF00'
        img = "http://maps.google.com/mapfiles/kml/pal2/icon5.png"
    }
    else if (cat=="2g."){
        col = '#00FF00'
        img = "http://maps.google.com/mapfiles/kml/shapes/gas_stations.png"
    }
    else if (cat=="3."){
        col = '#008080'
        img = "http://maps.google.com/mapfiles/kml/pal3/icon46.png"
    }
    else if (cat=="3a."){
        col = '#008080'
        img = "http://maps.google.com/mapfiles/kml/pal3/icon46.png"
    }
    else if (cat=="3b."){
        col = '#008080'
        img = "http://maps.google.com/mapfiles/kml/pal3/icon46.png"
    }
    else if (cat=="3d."){
        col = '#008080'
        img = "http://maps.google.com/mapfiles/kml/pal3/icon46.png"
    }
    else if (cat=="4."){
        col = '#F00080'
        img = "http://maps.google.com/mapfiles/kml/shapes/police.png"
    }
    else if (cat=="4a."){
        col = '#F00080'
        img = "http://maps.google.com/mapfiles/kml/shapes/police.png"
    }
    else if (cat=="5."){
        col = '#808000'
        img = "http://maps.google.com/mapfiles/kml/pal3/icon51.png"
    }
    else if (cat=="5a."){
        col = '#808000'
        img = "http://maps.google.com/mapfiles/kml/pal3/icon48.png"
    }
    else if (cat=="5b."){
        col = '#808000'
        img = "http://maps.google.com/mapfiles/kml/pal4/icon15.png"
    }
    else if (cat=="6."){
        col = '#8000F0'
        img = "http://maps.google.com/mapfiles/kml/shapes/earthquake.png"
    }
    else if (cat=="6a."){
        col = '#8000F0'
        img = "http://maps.google.com/mapfiles/kml/shapes/earthquake.png"
    }
    //else if (cat=="6c."){
    //    col = '#8000F0'
    //}
    //else if (cat=="6d."){
    //    col = '#8000F0'
    //}
    else if (cat=="7."){
        col = '#0000FF'
        img = "http://maps.google.com/mapfiles/kml/shapes/flag.png"
    }
    else if (cat=="7a."){
        col = '#0000FF'
        img = "http://maps.google.com/mapfiles/kml/pal5/icon11.png"
    }
    else if (cat=="7b."){
        col = '#0000FF'
        img = "http://maps.google.com/mapfiles/kml/pal4/icon3.png"
    }
    else if (cat=="7c."){
        col = '#0000FF'
        img = "http://maps.google.com/mapfiles/kml/shapes/pal4/icon4.png"
    }
    else if (cat=="7d."){
        col = '#0000FF'
        img = "http://maps.google.com/mapfiles/kml/shapes/truck.png"
    }
    else if (cat=="7e."){
        col = '#0000FF'
        img = "http://maps.google.com/mapfiles/kml/shapes/pal4/icon5.png"
    }
    else if (cat=="8."){
        col = '#505050'
        img = "http://maps.google.com/mapfiles/kml/shapes/info_circle.png"
    }
    else if (cat=="8a."){
        col = '#505050'
        img = "http://maps.google.com/mapfiles/kml/shapes/heliport.png"
    }
    else if (cat=="8b."){
        col = '#505050'
        img = "http://maps.google.com/mapfiles/kml/shapes/man.png"
    }
    else if (cat=="8c."){
        col = '#505050'
        img = "http://maps.google.com/mapfiles/kml/pal2/icon2.png"
    }
    else {
        alert("Unknown category: " + cat);
        col = '#000000'
        img = "http://maps.google.com/mapfiles/kml/shapes/ltblu-stars.png"
    }
    
    return [col,img];
}

function drawEvent(id, map, loc, size, cat, markers, openWindows){

    settings = getEventColour(eventCatIdx[id]);
    col = settings[0]
    img = settings[1]
    
    var lats = [];
    var lons = [];
    relReps = eventReports[id]
    for (var r in relReps){
        lats.push(reportLat[relReps[r]]);
        lons.push(reportLon[relReps[r]]);
    }

    var maxLat = Math.max.apply(null, lats);
    var minLat = Math.min.apply(null, lats);
    var maxLon = Math.max.apply(null, lons);
    var minLon = Math.min.apply(null, lons);
    
    distLat = Math.abs(maxLat - minLat);
    distLon = Math.abs(maxLon - minLon);
     
    avgDist = 10000.0 * (distLat + distLon)/2.0;
    
    opacity = 10*size / (avgDist^2);
//     
//     populationOptions["paths"] = lineCoords
//     eventMarking[0] = new google.maps.Polygon(populationOptions);        

    var populationOptions = {
        strokeColor: col,
        strokeOpacity: opacity,
        strokeWeight: 0,
        fillColor: col,
        fillOpacity: opacity,
        map: map,
        center: loc,
        radius: avgDist
    };
    eventMarking = new Array()
    // Add the circle for this city to the map.
    eventMarking[0] = new google.maps.Circle(populationOptions);
    

    marker = new google.maps.Marker({
        position: loc,
        map: map,
        title: 'Event!',
        icon: img
    });
    eventMarking[1] = marker
    markers[id] = eventMarking  
    
    conf = size / 20;
    if (conf > 0.99) conf = 0.99;
    
    google.maps.event.addListener(markers[id][1], 'click', function() {
                        
        var infoWindow = new google.maps.InfoWindow({
            content: '<b>' + cat + '<br\>Confidence: ' + conf + '<br/>Reports (ordered by relevance):</b>'
        });
        addEventInfo(infoWindow,id)
        
        infoWindow.open(map);//,markers[id][1]);
        infoWindow.setPosition(loc);
        for (var i in openWindows){
            
            openWindows[i].close();
            delete openWindows[i];
        }
        openWindows[id] = infoWindow;
    });
}

function loadScript() {
  var script = document.createElement('script');
  script.type = 'text/javascript';
  script.src = 'http://maps.googleapis.com/maps/api/js?key=AIzaSyAeqPac68BRWyDN4QSQDVU1RnkrIZVuhvc&sensor=false&' +
      'callback=initialize';
  document.body.appendChild(script);
}