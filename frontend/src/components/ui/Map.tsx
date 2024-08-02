import React from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { MapData, GeoJSONProperties } from '@/types';

interface MapProps {
  mapData: MapData;
  zoomLevel: number;
}

const Map: React.FC<MapProps> = ({ mapData, zoomLevel }) => {
  return (
    <MapContainer 
      center={[20.5937, 78.9629]} // Center of India
      zoom={5 * zoomLevel} 
      style={{ height: '400px', width: '100%' }}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      {mapData && (
        <GeoJSON 
          data={mapData.geojson}
          style={(feature) => ({
            fillColor: (feature?.properties as GeoJSONProperties)?.color,
            weight: 1,
            opacity: 1,
            color: 'white',
            fillOpacity: 0.7
          })}
        />
      )}
    </MapContainer>
  );
};

export default Map;