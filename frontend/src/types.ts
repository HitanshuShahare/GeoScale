export interface Option {
    value: string;
    label: string;
  }
  
  export interface MapData {
    geojson: GeoJSON.FeatureCollection;
  }
  
  export interface GeoJSONProperties {
    color: string;
    temperature: number;
  }