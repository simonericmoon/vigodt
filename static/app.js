var map = L.map('map').setView([49.80085320573547, 9.891294025945959], 15);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 25
}).addTo(map);

var icons = {
    car: L.icon({iconUrl: 'https://svgsilh.com/svg/309541.svg', iconSize: [38, 95]}),
    person: L.icon({iconUrl: 'https://svgsilh.com/svg/297255.svg', iconSize: [38, 95]}),
    truck: L.icon({iconUrl: 'https://svgsilh.com/svg/1918551.svg', iconSize: [38, 95]}),
    container: L.icon({iconUrl: 'https://svgsilh.com/svg/1576079.svg', iconSize: [38, 95]}),
    motorcycle: L.icon({iconUrl: 'https://svgsilh.com/svg/1131863.svg', iconSize: [38, 95]}),
    bus: L.icon({iconUrl: 'https://svgsilh.com/svg/296715.svg', iconSize: [38, 95]}),
    train: L.icon({iconUrl: 'https://www.svgrepo.com/show/115517/train.svg', iconSize: [38, 95]})

};

var allMarkers = []; // Array to store all markers, so we can filter them later! 

function updateConfidenceLabel() {
    var min = document.getElementById('confidenceMin').value;
    var max = document.getElementById('confidenceMax').value;
    document.getElementById('confidenceLabel').innerText = `${min} - ${max}`;
    filterMarkers();
}

function addMarker(obj) {
    var coordinates = obj.coordinates;
    var class_name = obj.class_name;
    var frameId = obj.frameId;
    var objectId = obj.objectId;
    var confidence = obj.confidence;
    
    var lat = coordinates[1];
    var lon = coordinates[0];
    var icon = icons[class_name] ? icons[class_name] : new L.Icon.Default();
    var marker = L.marker([lat, lon], {icon: icon})
                  .bindPopup('Class: ' + class_name + '<br>' +
                             'FrameID: ' + frameId + '<br>' +
                             'ObjectID: ' + objectId + '<br>' +
                             'Confidence: ' + confidence.toFixed(2));
    marker.addTo(map);
    allMarkers.push({marker, class_name, confidence, frameId}); // Include frameId here
}


function filterMarkers() {
    allMarkers.forEach(({marker, class_name, confidence, frameId}) => {
        var selectedType = document.getElementById('objectType').value;
        var minConfidence = parseFloat(document.getElementById('confidenceMin').value);
        var maxConfidence = parseFloat(document.getElementById('confidenceMax').value);
        var filterFrameId = document.getElementById('frameId').value.trim();

        if ((class_name === selectedType || selectedType === 'all') && 
            confidence >= minConfidence && confidence <= maxConfidence &&
            (filterFrameId === '' || frameId.toString() === filterFrameId)) {
            marker.addTo(map);
        } else {
            map.removeLayer(marker);
        }
    });
}



document.getElementById('frameId').addEventListener('input', filterMarkers);

document.getElementById('confidenceMin').addEventListener('input', updateConfidenceLabel);
document.getElementById('confidenceMax').addEventListener('input', updateConfidenceLabel);

fetch('/data')
    .then(response => response.json())
    .then(data => {
        data.forEach(obj => {
            addMarker(obj);
        });
        filterMarkers(); 
    });
